import { describe, expect } from "bun:test"
import { Effect, Layer } from "effect"
import { HttpClientRequest } from "effect/unstable/http"
import { Image, ImageClient } from "../../src"
import { XAI } from "../../src/providers"
import { it } from "../lib/effect"
import { dynamicResponse } from "../lib/http"

describe("xAI Images", () => {
  it.effect("generates through the OpenAI-compatible Images API", () =>
    Effect.gen(function* () {
      const response = yield* Image.generate({
        model: XAI.configure({
          apiKey: "test",
          baseURL: "https://api.xai.test/v1",
          image: { providerOptions: { resolution: "1k", responseFormat: "b64_json" } },
        }).image("grok-imagine-image"),
        prompt: "A robot tending a rooftop garden",
        count: 2,
        aspectRatio: "16:9",
      })

      expect(response.images).toHaveLength(2)
      expect(response.image?.mediaType).toBe("image/jpeg")
      expect(response.image?.data).toEqual(Uint8Array.from([1, 2, 3]))
      expect(response.providerMetadata).toEqual({ xai: { usage: { num_images: 2 } } })
    }).pipe(
      Effect.provide(
        ImageClient.layer.pipe(
          Layer.provide(
            dynamicResponse((input) =>
              Effect.gen(function* () {
                const request = yield* HttpClientRequest.toWeb(input.request).pipe(Effect.orDie)
                expect(request.url).toBe("https://api.xai.test/v1/images/generations")
                expect(request.headers.get("authorization")).toBe("Bearer test")
                expect(JSON.parse(input.text)).toEqual({
                  model: "grok-imagine-image",
                  prompt: "A robot tending a rooftop garden",
                  n: 2,
                  aspect_ratio: "16:9",
                  resolution: "1k",
                  response_format: "b64_json",
                })
                return input.respond(
                  JSON.stringify({
                    data: [
                      { b64_json: "AQID", mime_type: "image/jpeg" },
                      { b64_json: "BAUG", mime_type: "image/jpeg" },
                    ],
                    usage: { num_images: 2 },
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
})
