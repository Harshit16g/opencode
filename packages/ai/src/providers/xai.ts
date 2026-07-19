import { AuthOptions, type ProviderAuthOption } from "../route/auth-options"
import type { RouteDefaultsInput } from "../route/client"
import { HttpOptions, ProviderID, type ModelID } from "../schema"
import * as OpenAICompatibleProfiles from "./openai-compatible-profile"
import * as OpenAICompatibleChat from "../protocols/openai-compatible-chat"
import * as OpenAIResponses from "../protocols/openai-responses"
import { OpenAIImages, type XAIImageOptions } from "../protocols/openai-images"

export const id = ProviderID.make("xai")

export type ModelOptions = RouteDefaultsInput &
  ProviderAuthOption<"optional"> & {
    readonly baseURL?: string
    readonly image?: ImageConfig
  }

export interface ImageConfig {
  readonly providerOptions?: XAIImageOptions
}

export type { XAIImageOptions } from "../protocols/openai-images"

export const routes = [OpenAIResponses.route, OpenAICompatibleChat.route]

const auth = (options: ProviderAuthOption<"optional">) => AuthOptions.bearer(options, "XAI_API_KEY")

const configuredResponsesRoute = (input: ModelOptions) => {
  const { apiKey: _, auth: _auth, baseURL, image: _image, ...rest } = input
  return OpenAIResponses.route.with({
    ...rest,
    provider: id,
    endpoint: { baseURL: baseURL ?? OpenAICompatibleProfiles.profiles.xai.baseURL },
    auth: auth(input),
  })
}

const configuredChatRoute = (input: ModelOptions) => {
  const { apiKey: _, auth: _auth, baseURL, image: _image, ...rest } = input
  return OpenAICompatibleChat.route.with({
    ...rest,
    provider: id,
    endpoint: { baseURL: baseURL ?? OpenAICompatibleProfiles.profiles.xai.baseURL },
    auth: auth(input),
  })
}

export const configure = (input: ModelOptions = {}) => {
  const responsesRoute = configuredResponsesRoute(input)
  const chatRoute = configuredChatRoute(input)
  const responses = (modelID: string | ModelID) => responsesRoute.model({ id: modelID })
  const chat = (modelID: string | ModelID) => chatRoute.model({ id: modelID })
  const image = (modelID: string | ModelID) =>
    OpenAIImages.model({
      id: modelID,
      protocol: "xai",
      auth: auth(input),
      baseURL: input.baseURL ?? OpenAICompatibleProfiles.profiles.xai.baseURL,
      headers: input.headers,
      defaults: {
        providerOptions:
          input.image?.providerOptions === undefined ? undefined : { xai: { ...input.image.providerOptions } },
        http: input.http === undefined ? undefined : HttpOptions.make(input.http),
      },
    })
  return {
    id,
    model: responses,
    responses,
    chat,
    image,
    configure,
  }
}

export const provider = configure()
export const model = provider.model
export const responses = provider.responses
export const chat = provider.chat
export const image = provider.image
