import { Effect, Encoding, Schema } from "effect"
import { Headers, HttpClientRequest } from "effect/unstable/http"
import {
  ImageModel,
  GeneratedImage,
  ImageResponse,
  type ImageRequest,
  type ImageModelDefaults,
  type ImageRoute,
} from "../image"
import { Auth, type Definition as AuthDefinition } from "../route/auth"
import { InvalidProviderOutputReason, LLMError, Usage, mergeHttpOptions, mergeJsonRecords } from "../schema"
import { ProviderShared } from "./shared"
import { OpenAIImage } from "./utils/openai-image"

const ADAPTER = "openai-images"
export const DEFAULT_BASE_URL = "https://api.openai.com/v1"
export const PATH = "/images/generations"

export interface OpenAIImageOptions {
  readonly quality?: "auto" | "low" | "medium" | "high"
  readonly background?: "auto" | "opaque" | "transparent"
  readonly moderation?: "auto" | "low"
  readonly outputFormat?: "png" | "jpeg" | "webp"
  readonly outputCompression?: number
}

export interface ZAIImageOptions {
  readonly quality?: "hd" | "standard"
  readonly userID?: string
}

export type ImageProtocol = "openai" | "zai"

const OpenAIImageBody = Schema.Struct({
  model: Schema.String,
  prompt: Schema.String,
  n: Schema.optional(Schema.Int.check(Schema.isGreaterThanOrEqualTo(1))),
  size: Schema.optional(Schema.String),
  quality: Schema.optional(Schema.Literals(["auto", "low", "medium", "high"])),
  background: Schema.optional(Schema.Literals(["auto", "opaque", "transparent"])),
  moderation: Schema.optional(Schema.Literals(["auto", "low"])),
  output_format: Schema.optional(Schema.Literals(["png", "jpeg", "webp"])),
  output_compression: Schema.optional(Schema.Int.check(Schema.isBetween({ minimum: 0, maximum: 100 }))),
})
export type OpenAIImageBody = Schema.Schema.Type<typeof OpenAIImageBody>

const ZAIImageBody = Schema.Struct({
  model: Schema.String,
  prompt: Schema.String,
  size: Schema.optional(Schema.String),
  quality: Schema.optional(Schema.Literals(["hd", "standard"])),
  user_id: Schema.optional(Schema.String.check(Schema.isMinLength(6), Schema.isMaxLength(128))),
})
export type ZAIImageBody = Schema.Schema.Type<typeof ZAIImageBody>

const OpenAIImageResponse = Schema.Struct({
  data: Schema.Array(
    Schema.Struct({
      b64_json: Schema.optional(Schema.String),
      url: Schema.optional(Schema.String),
      revised_prompt: Schema.optional(Schema.String),
    }),
  ),
  output_format: Schema.optional(Schema.String),
  usage: Schema.optional(
    Schema.Struct({
      input_tokens: Schema.optional(Schema.Number),
      output_tokens: Schema.optional(Schema.Number),
      total_tokens: Schema.optional(Schema.Number),
      input_tokens_details: Schema.optional(Schema.Record(Schema.String, Schema.Unknown)),
      output_tokens_details: Schema.optional(Schema.Record(Schema.String, Schema.Unknown)),
    }),
  ),
})

const ZAIImageResponse = Schema.Struct({
  created: Schema.optional(Schema.Int),
  id: Schema.optional(Schema.String),
  request_id: Schema.optional(Schema.String),
  data: Schema.Array(Schema.Struct({ url: Schema.String })),
  content_filter: Schema.optional(
    Schema.Array(
      Schema.Struct({
        role: Schema.optional(Schema.Literals(["assistant", "user", "history"])),
        level: Schema.optional(Schema.Int.check(Schema.isBetween({ minimum: 0, maximum: 3 }))),
      }),
    ),
  ),
})

export interface ModelInput {
  readonly id: string
  readonly protocol?: ImageProtocol
  readonly auth: AuthDefinition
  readonly baseURL?: string
  readonly headers?: Record<string, string>
  readonly defaults?: ImageModelDefaults
}

const providerOptions = (request: ImageRequest): OpenAIImageOptions => ({
  ...request.model.defaults?.providerOptions?.openai,
  ...request.providerOptions?.openai,
})

const zaiOptions = (request: ImageRequest): ZAIImageOptions => ({
  ...request.model.defaults?.providerOptions?.zai,
  ...request.providerOptions?.zai,
})

const openaiBody = (request: ImageRequest): OpenAIImageBody => {
  const options = providerOptions(request)
  return {
    model: request.model.id,
    prompt: request.prompt,
    n: request.count,
    size: request.size === undefined ? undefined : `${request.size.width}x${request.size.height}`,
    quality: options.quality,
    background: options.background,
    moderation: options.moderation,
    output_format: options.outputFormat,
    output_compression: options.outputCompression,
  }
}

const zaiBody = (request: ImageRequest): ZAIImageBody => {
  const options = zaiOptions(request)
  return {
    model: request.model.id,
    prompt: request.prompt,
    size: request.size === undefined ? undefined : `${request.size.width}x${request.size.height}`,
    quality: options.quality,
    user_id: options.userID,
  }
}

const invalidOutput = (adapter: string, message: string) =>
  new LLMError({
    module: adapter,
    method: "generate",
    reason: new InvalidProviderOutputReason({ message, route: adapter }),
  })

const applyQuery = (url: string, query: Record<string, string> | undefined) => {
  if (!query) return url
  const next = new URL(url)
  Object.entries(query).forEach(([key, value]) => next.searchParams.set(key, value))
  return next.toString()
}

const OPENAI_BODY_FIELDS = new Set([
  "model",
  "prompt",
  "n",
  "size",
  "quality",
  "background",
  "moderation",
  "output_format",
  "output_compression",
])
const ZAI_BODY_FIELDS = new Set(["model", "prompt", "size", "quality", "user_id"])

const bodyWithOverlay = Effect.fn("OpenAIImages.bodyWithOverlay")(function* (
  imageBody: Record<string, unknown>,
  overlay: Record<string, unknown> | undefined,
  ownedFields: ReadonlySet<string>,
) {
  if (!overlay) return imageBody
  const reserved = Object.keys(overlay).filter((key) => ownedFields.has(key))
  if (reserved.length > 0)
    return yield* ProviderShared.invalidRequest(
      `http.body cannot overlay protocol-owned field(s): ${reserved.join(", ")}`,
    )
  return mergeJsonRecords(imageBody, overlay) ?? imageBody
})

export const model = (input: ModelInput) => {
  const protocol = input.protocol ?? "openai"
  const adapter = protocol === "openai" ? ADAPTER : "zai-images"
  const name = protocol === "openai" ? "OpenAI" : "Z.ai"
  const route: ImageRoute = {
    id: adapter,
    generate: Effect.fn("OpenAIImages.generate")(function* (request: ImageRequest, execute) {
      if (request.aspectRatio !== undefined)
        return yield* ProviderShared.invalidRequest(`${name} Images does not support the common aspectRatio option`)
      if (request.seed !== undefined)
        return yield* ProviderShared.invalidRequest(`${name} Images does not support the common seed option`)
      if (protocol === "zai" && request.count !== undefined)
        return yield* ProviderShared.invalidRequest("Z.ai Images does not support the common count option")

      const requestBody =
        protocol === "openai"
          ? yield* ProviderShared.validateWith(Schema.decodeUnknownEffect(OpenAIImageBody))(openaiBody(request))
          : yield* ProviderShared.validateWith(Schema.decodeUnknownEffect(ZAIImageBody))(zaiBody(request))
      const http = mergeHttpOptions(request.model.defaults?.http, request.http)
      const overlaidBody = yield* bodyWithOverlay(
        requestBody,
        http?.body,
        protocol === "openai" ? OPENAI_BODY_FIELDS : ZAI_BODY_FIELDS,
      )
      const text = ProviderShared.encodeJson(overlaidBody)
      const url = applyQuery(`${(input.baseURL ?? DEFAULT_BASE_URL).replace(/\/$/, "")}${PATH}`, http?.query)
      const headers = yield* Auth.toEffect(input.auth)({
        request,
        method: "POST",
        url,
        body: text,
        headers: Headers.fromInput({ ...input.headers, ...http?.headers }),
      })
      const response = yield* execute(
        HttpClientRequest.post(url).pipe(
          HttpClientRequest.setHeaders(headers),
          HttpClientRequest.bodyText(text, "application/json"),
        ),
      )
      const payload = yield* response.json.pipe(
        Effect.mapError(() => invalidOutput(adapter, `Failed to read the ${name} Images response`)),
      )
      const decoded = yield* (
        protocol === "openai"
          ? Schema.decodeUnknownEffect(OpenAIImageResponse)(payload)
          : Schema.decodeUnknownEffect(ZAIImageResponse)(payload)
      ).pipe(Effect.mapError(() => invalidOutput(adapter, `${name} Images returned an invalid response`)))
      const format =
        protocol === "zai"
          ? "jpeg"
          : (("output_format" in decoded ? decoded.output_format : undefined) ??
            providerOptions(request).outputFormat ??
            "png")
      const images = yield* Effect.forEach(decoded.data, (item, index) => {
        if ("b64_json" in item && item.b64_json)
          return Effect.fromResult(Encoding.decodeBase64(item.b64_json)).pipe(
            Effect.mapError(() =>
              invalidOutput(adapter, `${name} Images result ${index} contains invalid base64 data`),
            ),
            Effect.map(
              (data) =>
                new GeneratedImage({
                  mediaType: `image/${format}`,
                  data,
                  providerMetadata:
                    "revised_prompt" in item && item.revised_prompt !== undefined
                      ? { openai: { revisedPrompt: item.revised_prompt } }
                      : undefined,
                }),
            ),
          )
        if (item.url)
          return Effect.succeed(
            new GeneratedImage({
              mediaType: `image/${format}`,
              data: item.url,
              providerMetadata:
                "revised_prompt" in item && item.revised_prompt !== undefined
                  ? { openai: { revisedPrompt: item.revised_prompt } }
                  : undefined,
            }),
          )
        return Effect.fail(invalidOutput(adapter, `${name} Images result ${index} has neither image data nor a URL`))
      })
      if (images.length === 0) return yield* invalidOutput(adapter, `${name} Images returned no images`)
      return new ImageResponse({
        images,
        usage:
          protocol === "zai" || !("usage" in decoded) || decoded.usage === undefined
            ? undefined
            : new Usage({
                inputTokens: decoded.usage.input_tokens,
                outputTokens: decoded.usage.output_tokens,
                totalTokens: decoded.usage.total_tokens,
                providerMetadata: { openai: decoded.usage },
              }),
        providerMetadata:
          protocol === "openai"
            ? { openai: { outputFormat: format } }
            : {
                zai: {
                  created: "created" in decoded ? decoded.created : undefined,
                  id: "id" in decoded ? decoded.id : undefined,
                  requestID: "request_id" in decoded ? decoded.request_id : undefined,
                  contentFilter: "content_filter" in decoded ? decoded.content_filter : undefined,
                },
              },
      })
    }),
  }
  return ImageModel.make({ id: input.id, provider: protocol, route, defaults: input.defaults })
}

export const OpenAIImages = {
  model,
} as const
