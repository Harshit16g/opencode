import { describe, expect } from "bun:test"
import { Effect, Layer } from "effect"
import { HttpClientRequest } from "effect/unstable/http"
import { Image, ImageClient } from "../../src"
import { OpenAI, ZAI } from "../../src/providers"
import { it } from "../lib/effect"
import { dynamicResponse, fixedResponse } from "../lib/http"

describe("Z.ai Images", () => {
  it.effect("generates through the Z.ai Images API", () =>
    Effect.gen(function* () {
      const response = yield* Image.generate({
        model: ZAI.configure({
          apiKey: "test",
          baseURL: "https://api.z.ai.test/api/paas/v4",
          headers: { "x-default": "yes" },
          http: { body: { request_metadata: "value" }, query: { trace: "default" } },
          image: { providerOptions: { quality: "standard", userID: "user-123" } },
        }).image("glm-image"),
        prompt: "A red circle on a white background",
        size: { width: 1280, height: 1280 },
        providerOptions: { zai: { quality: "hd" } },
        http: { headers: { "x-request": "yes" }, query: { trace: "request" } },
      })

      expect(response.images).toHaveLength(1)
      expect(response.image?.mediaType).toBe("image/jpeg")
      expect(response.image?.data).toBe("https://cdn.z.ai/generated.png")
      expect(response.providerMetadata).toEqual({
        zai: {
          created: 1_760_335_349,
          id: "generation-1",
          requestID: "request-1",
          contentFilter: [{ role: "assistant", level: 3 }],
        },
      })
    }).pipe(
      Effect.provide(
        ImageClient.layer.pipe(
          Layer.provide(
            dynamicResponse((input) =>
              Effect.gen(function* () {
                const request = yield* HttpClientRequest.toWeb(input.request).pipe(Effect.orDie)
                expect(request.url).toBe("https://api.z.ai.test/api/paas/v4/images/generations?trace=request")
                expect(request.headers.get("authorization")).toBe("Bearer test")
                expect(request.headers.get("x-default")).toBe("yes")
                expect(request.headers.get("x-request")).toBe("yes")
                expect(JSON.parse(input.text)).toEqual({
                  model: "glm-image",
                  prompt: "A red circle on a white background",
                  size: "1280x1280",
                  quality: "hd",
                  user_id: "user-123",
                  request_metadata: "value",
                })
                return input.respond(
                  JSON.stringify({
                    created: 1_760_335_349,
                    id: "generation-1",
                    request_id: "request-1",
                    data: [{ url: "https://cdn.z.ai/generated.png" }],
                    content_filter: [{ role: "assistant", level: 3 }],
                  }),
                  { headers: { "content-type": "application/json" } },
                )
              }),
            ),
          ),
        ),
      ),
    ),
  )

  it.effect("validates Z.ai-owned request fields without reserving them for OpenAI", () =>
    Effect.gen(function* () {
      const invalid = yield* Image.generate({
        model: ZAI.configure({ apiKey: "test", image: { providerOptions: { userID: "short" } } }).image("model"),
        prompt: "test",
      }).pipe(Effect.provide(ImageClient.layer.pipe(Layer.provide(fixedResponse("{}")))), Effect.flip)
      expect(invalid.reason._tag).toBe("InvalidRequest")

      const openaiQuality = yield* Image.generate({
        model: OpenAI.configure({ apiKey: "test" }).image("model"),
        prompt: "test",
        providerOptions: { openai: { quality: "standard" } },
      }).pipe(Effect.provide(ImageClient.layer.pipe(Layer.provide(fixedResponse("{}")))), Effect.flip)
      expect(openaiQuality.reason._tag).toBe("InvalidRequest")

      const zaiOverlay = yield* Image.generate({
        model: ZAI.configure({ apiKey: "test" }).image("model"),
        prompt: "test",
        http: { body: { user_id: "overlay-user" } },
      }).pipe(Effect.provide(ImageClient.layer.pipe(Layer.provide(fixedResponse("{}")))), Effect.flip)
      expect(zaiOverlay.reason).toMatchObject({
        _tag: "InvalidRequest",
        message: "http.body cannot overlay protocol-owned field(s): user_id",
      })

      const request = yield* Image.generate({
        model: OpenAI.configure({ apiKey: "test" }).image("model"),
        prompt: "test",
        http: { body: { user_id: "overlay-user" } },
      }).pipe(
        Effect.provide(
          ImageClient.layer.pipe(
            Layer.provide(
              dynamicResponse((input) => {
                expect(JSON.parse(input.text)).toMatchObject({ user_id: "overlay-user" })
                return Effect.succeed(
                  input.respond(JSON.stringify({ data: [{ url: "https://example.test/image.jpg" }] }), {
                    headers: { "content-type": "application/json" },
                  }),
                )
              }),
            ),
          ),
        ),
      )
      expect(request.image?.data).toBe("https://example.test/image.jpg")
    }),
  )

  it.effect("rejects invalid Z.ai content filter structures", () =>
    Effect.gen(function* () {
      const model = ZAI.configure({ apiKey: "test" }).image("model")
      const payloads = [
        { data: [{ url: "https://example.test/image.jpg" }], content_filter: [{ role: "system", level: 1 }] },
        { data: [{ url: "https://example.test/image.jpg" }], content_filter: [{ role: "user", level: 1.5 }] },
        { data: [{ url: "https://example.test/image.jpg" }], content_filter: [{ role: "history", level: 4 }] },
      ]

      yield* Effect.forEach(payloads, (payload) =>
        Image.generate({ model, prompt: "test" }).pipe(
          Effect.provide(
            ImageClient.layer.pipe(
              Layer.provide(
                fixedResponse(JSON.stringify(payload), { headers: { "content-type": "application/json" } }),
              ),
            ),
          ),
          Effect.flip,
          Effect.tap((error) => Effect.sync(() => expect(error.reason._tag).toBe("InvalidProviderOutput"))),
        ),
      )
    }),
  )
})
