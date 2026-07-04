import Foundation

public struct OpenAIErrorResponse: Sendable, Equatable {
    public let status: Int
    public let body: [String: String?]
    public let retryAfterSeconds: Int?

    public init(status: Int, message: String, retryAfterSeconds: Int? = nil) {
        self.status = status
        self.body = [
            "message": message,
            "type": openAIErrorType(status: status),
            "param": nil,
            "code": nil,
        ]
        self.retryAfterSeconds = retryAfterSeconds
    }
}

public func openAIErrorResponse(
    message: String,
    status: Int,
    retryAfterSeconds: Int? = nil
) -> OpenAIErrorResponse {
    OpenAIErrorResponse(status: status, message: message, retryAfterSeconds: retryAfterSeconds)
}

public func openAIErrorBody(
    message: String,
    status: Int,
    param: Any? = nil,
    code: Any? = nil
) -> [String: Any] {
    ["error": openAIErrorObject(message: message, status: status, param: param, code: code)]
}

public func openAIErrorObject(
    message: String,
    status: Int,
    param: Any? = nil,
    code: Any? = nil
) -> [String: Any] {
    [
        "message": message,
        "type": openAIErrorType(status: status),
        "param": param ?? NSNull(),
        "code": code ?? NSNull(),
    ]
}

public func openAIErrorType(status: Int) -> String {
    switch status {
    case 401:
        return "authentication_error"
    case 404:
        return "not_found_error"
    case 413:
        return "invalid_request_error"
    case 429:
        return "rate_limit_error"
    case 500...:
        return "server_error"
    default:
        return "invalid_request_error"
    }
}
