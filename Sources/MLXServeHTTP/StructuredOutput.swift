import Foundation

/// Structured output modes observable at the HTTP layer.
///
/// Only `.choice` can be implemented as true constrained decoding in the native
/// sampler. JSON object/schema modes intentionally represent the omlx fallback:
/// prompt injection plus a Warning response header when grammar compilation is
/// unavailable. Grammar and regex requests fail at parse time because this port
/// does not bind xgrammar.
public enum StructuredOutputSpec: Sendable, Equatable {
    case none
    case jsonObject
    case jsonSchema(name: String?, schema: [String: OpenAIJSONValue])
    case choice([String])

    public var warningMessage: String? {
        switch self {
        case .jsonObject:
            return "structured output json_object uses prompt-injection fallback; grammar-constrained decoding is unavailable"
        case .jsonSchema:
            return "structured output json_schema uses prompt-injection fallback; grammar-constrained decoding is unavailable"
        case .none, .choice:
            return nil
        }
    }
}

enum StructuredOutputParser {
    static func parse(from object: [String: Any]) throws -> StructuredOutputSpec {
        if hasNonNullValue(object["guided_grammar"]) {
            throw OpenAIServerError.invalidStructuredOutput("guided_grammar requires grammar compilation, which is unavailable")
        }

        // omlx-specific guided controls are more explicit than OpenAI response_format.
        if let rawStructuredOutputs = object["structured_outputs"], !(rawStructuredOutputs is NSNull) {
            return try parseStructuredOutputs(rawStructuredOutputs)
        }

        if let rawResponseFormat = object["response_format"], !(rawResponseFormat is NSNull) {
            return try parseResponseFormat(rawResponseFormat)
        }

        return .none
    }

    private static func parseResponseFormat(_ value: Any) throws -> StructuredOutputSpec {
        guard let object = value as? [String: Any],
            let type = object["type"] as? String
        else {
            throw OpenAIServerError.invalidJSON
        }

        switch type {
        case "text":
            return .none
        case "json_object":
            return .jsonObject
        case "json_schema":
            guard let jsonSchema = object["json_schema"] as? [String: Any],
                let rawSchema = jsonSchema["schema"] as? [String: Any]
            else {
                throw OpenAIServerError.invalidJSON
            }
            return .jsonSchema(
                name: jsonSchema["name"] as? String,
                schema: try convertJSONObject(rawSchema)
            )
        default:
            throw OpenAIServerError.invalidJSON
        }
    }

    private static func parseStructuredOutputs(_ value: Any) throws -> StructuredOutputSpec {
        guard let object = value as? [String: Any],
            let type = object["type"] as? String
        else {
            throw OpenAIServerError.invalidJSON
        }

        switch type {
        case "json":
            return .jsonObject
        case "choice":
            guard let choices = object["choices"] as? [String] else {
                throw OpenAIServerError.invalidJSON
            }
            guard !choices.isEmpty else {
                throw OpenAIServerError.invalidStructuredOutput("structured_outputs.choice requires at least one choice")
            }
            return .choice(choices)
        case "grammar", "regex":
            throw OpenAIServerError.invalidStructuredOutput(
                "structured_outputs.\(type) requires grammar compilation, which is unavailable"
            )
        default:
            throw OpenAIServerError.invalidJSON
        }
    }

    private static func convertJSONObject(_ object: [String: Any]) throws -> [String: OpenAIJSONValue] {
        var converted: [String: OpenAIJSONValue] = [:]
        for (key, value) in object {
            guard let jsonValue = OpenAIJSONValue(value) else {
                throw OpenAIServerError.invalidJSON
            }
            converted[key] = jsonValue
        }
        return converted
    }

    private static func hasNonNullValue(_ value: Any?) -> Bool {
        guard let value else { return false }
        return !(value is NSNull)
    }
}
