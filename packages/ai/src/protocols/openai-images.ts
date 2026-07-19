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

export const DEFAULT_BASE_URL = "https://api.openai.com/v1"
export const PATH = "/images/generations"

export interface OpenAIImageOptions {
  readonly quality?: "auto" | "low" | "medium" | "high"
  readonly background?: "auto" | "opaque" | "transparent"
  readonly moderation?: "auto" | "low"
  readonly outputFormat?: "png" | "jpeg" | "webp"
  readonly outputCompression?: number
}

export interface XAIImageOptions {
  readonly resolution?: "1k" | "2k"
  readonly responseFormat?: "url" | "b64_json"
}

export type ImageProtocol = "openai" | "xai"

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
  aspect_ratio: Schema.optional(Schema.String),
  resolution: Schema.optional(Schema.Literals(["1k", "2k"])),
  response_format: Schema.optional(Schema.Literals(["url", "b64_json"])),
})
export type OpenAIImageBody = Schema.Schema.Type<typeof OpenAIImageBody>

const OpenAIImageResponse = Schema.Struct({
  data: Schema.Array(
    Schema.Struct({
      b64_json: Schema.optional(Schema.String),
      url: Schema.optional(Schema.String),
      revised_prompt: Schema.optional(Schema.String),
      mime_type: Schema.optional(Schema.String),
    }),
  ),
  output_format: Schema.optional(Schema.String),
  usage: Schema.optional(Schema.Unknown),
})

const OpenAIImageUsage = Schema.Struct({
  input_tokens: Schema.optional(Schema.Number),
  output_tokens: Schema.optional(Schema.Number),
  total_tokens: Schema.optional(Schema.Number),
  input_tokens_details: Schema.optional(Schema.Record(Schema.String, Schema.Unknown)),
  output_tokens_details: Schema.optional(Schema.Record(Schema.String, Schema.Unknown)),
})

export interface ModelInput {
  readonly id: string
  readonly protocol?: ImageProtocol
  readonly auth: AuthDefinition
  readonly baseURL?: string
  readonly headers?: Record<string, string>
  readonly defaults?: ImageModelDefaults
}

const openAIOptions = (request: ImageRequest): OpenAIImageOptions => ({
  ...request.model.defaults?.providerOptions?.openai,
  ...request.providerOptions?.openai,
})

const xaiOptions = (request: ImageRequest): XAIImageOptions => ({
  ...request.model.defaults?.providerOptions?.xai,
  ...request.providerOptions?.xai,
})

const body = (request: ImageRequest, protocol: ImageProtocol): OpenAIImageBody => {
  if (protocol === "xai") {
    const options = xaiOptions(request)
    return {
      model: request.model.id,
      prompt: request.prompt,
      n: request.count,
      aspect_ratio: request.aspectRatio,
      resolution: options.resolution,
      response_format: options.responseFormat,
    }
  }
  const options = openAIOptions(request)
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

const PROTOCOL_BODY_FIELDS = new Set([
  "model",
  "prompt",
  "n",
  "size",
  "quality",
  "background",
  "moderation",
  "output_format",
  "output_compression",
  "aspect_ratio",
  "resolution",
  "response_format",
])

const bodyWithOverlay = Effect.fn("OpenAIImages.bodyWithOverlay")(function* (
  imageBody: OpenAIImageBody,
  overlay: Record<string, unknown> | undefined,
) {
  if (!overlay) return imageBody
  const reserved = Object.keys(overlay).filter((key) => PROTOCOL_BODY_FIELDS.has(key))
  if (reserved.length > 0)
    return yield* ProviderShared.invalidRequest(
      `http.body cannot overlay protocol-owned field(s): ${reserved.join(", ")}`,
    )
  return mergeJsonRecords(imageBody, overlay) ?? imageBody
})

export const model = (input: ModelInput) => {
  const protocol = input.protocol ?? "openai"
  const adapter = `${protocol}-images`
  const route: ImageRoute = {
    id: adapter,
    generate: Effect.fn("OpenAIImages.generate")(function* (request: ImageRequest, execute) {
      if (protocol === "openai" && request.aspectRatio !== undefined)
        return yield* ProviderShared.invalidRequest("OpenAI Images does not support the common aspectRatio option")
      if (protocol === "xai" && request.size !== undefined)
        return yield* ProviderShared.invalidRequest("xAI Images does not support the common size option")
      if (request.seed !== undefined)
        return yield* ProviderShared.invalidRequest(
          `${protocol === "openai" ? "OpenAI" : "xAI"} Images does not support the common seed option`,
        )

      const requestBody = yield* ProviderShared.validateWith(Schema.decodeUnknownEffect(OpenAIImageBody))(
        body(request, protocol),
      )
      const http = mergeHttpOptions(request.model.defaults?.http, request.http)
      const overlaidBody = yield* bodyWithOverlay(requestBody, http?.body)
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
        Effect.mapError(() => invalidOutput(adapter, `Failed to read the ${protocol} Images response`)),
      )
      const decoded = yield* Schema.decodeUnknownEffect(OpenAIImageResponse)(payload).pipe(
        Effect.mapError(() => invalidOutput(adapter, `${protocol} Images returned an invalid response`)),
      )
      const format = decoded.output_format ?? openAIOptions(request).outputFormat ?? "png"
      const images = yield* Effect.forEach(decoded.data, (item, index) => {
        const mediaType = item.mime_type ?? `image/${format}`
        if (item.b64_json)
          return Effect.fromResult(Encoding.decodeBase64(item.b64_json)).pipe(
            Effect.mapError(() =>
              invalidOutput(adapter, `${protocol} Images result ${index} contains invalid base64 data`),
            ),
            Effect.map(
              (data) =>
                new GeneratedImage({
                  mediaType,
                  data,
                  providerMetadata:
                    item.revised_prompt === undefined
                      ? undefined
                      : { [protocol]: { revisedPrompt: item.revised_prompt } },
                }),
            ),
          )
        if (item.url)
          return Effect.succeed(
            new GeneratedImage({
              mediaType,
              data: item.url,
              providerMetadata:
                item.revised_prompt === undefined ? undefined : { [protocol]: { revisedPrompt: item.revised_prompt } },
            }),
          )
        return Effect.fail(
          invalidOutput(adapter, `${protocol} Images result ${index} has neither image data nor a URL`),
        )
      })
      if (images.length === 0) return yield* invalidOutput(adapter, `${protocol} Images returned no images`)
      const usage = protocol === "openai" && Schema.is(OpenAIImageUsage)(decoded.usage) ? decoded.usage : undefined
      return new ImageResponse({
        images,
        usage:
          usage === undefined
            ? undefined
            : new Usage({
                inputTokens: usage.input_tokens,
                outputTokens: usage.output_tokens,
                totalTokens: usage.total_tokens,
                providerMetadata: { [protocol]: usage },
              }),
        providerMetadata: {
          [protocol]: {
            ...(protocol === "openai" ? { outputFormat: format } : {}),
            ...(protocol === "xai" && decoded.usage !== undefined ? { usage: decoded.usage } : {}),
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
