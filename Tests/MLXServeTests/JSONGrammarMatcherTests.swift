import MLX
@testable import MLXServe
import XCTest

final class JSONGrammarMatcherTests: XCTestCase {
    func testAllowedTokenIDsMatchesBruteForceAcceptanceAcrossStates() {
        let configuration = JSONGrammarConfiguration(tokens: Self.tokens, schema: .jsonObject)
        let matcher = configuration.makeMatcher()

        // Walk through empty → open → complete states; at each state the bucket-pruned
        // allowed set must equal a brute-force accepts() sweep over the whole vocabulary.
        for step in [nil, "{", "}"] as [String?] {
            if let step {
                matcher.advance(tokenID: Self.id(step))
            }
            let bruteForce = Set(
                Self.tokens.filter { matcher.accepts(tokenID: $0.id) }.map(\.id)
            )
            XCTAssertEqual(Set(matcher.allowedTokenIDs()), bruteForce)
        }
        XCTAssertTrue(matcher.allowedTokenIDs().contains(Self.eosID))
    }

    func testGrammarMaskConstrainsGreedyArgmaxWhenCandidateIsInvalid() throws {
        try MLXMetalRuntime.requireAvailable()

        let matcher = JSONGrammarConfiguration(tokens: Self.tokens, schema: .jsonObject)
            .makeMatcher()
        // Highest logit is "[" (id 3) — invalid for a top-level object; the rejection
        // path must fall back to the mask and pick "{" (id 1), the valid runner-up.
        var logits = [Float](repeating: -10, count: 20)
        logits[3] = 99
        logits[1] = 5
        let sampled = TokenSampler.sample(
            logits: MLXArray(logits),
            parameters: SamplingParameters(temperature: 0),
            jsonGrammarMatcher: matcher
        )

        XCTAssertEqual(sampled.item(Int.self), 1)
    }

    func testGrammarFastPathKeepsValidGreedyCandidate() throws {
        try MLXMetalRuntime.requireAvailable()

        let matcher = JSONGrammarConfiguration(tokens: Self.tokens, schema: .jsonObject)
            .makeMatcher()
        var logits = [Float](repeating: -10, count: 20)
        logits[1] = 99
        let sampled = TokenSampler.sample(
            logits: MLXArray(logits),
            parameters: SamplingParameters(temperature: 0),
            jsonGrammarMatcher: matcher
        )

        XCTAssertEqual(sampled.item(Int.self), 1)
    }

    func testJSONObjectAllowsOnlyObjectPrefixesAndGatesEOS() {
        let matcher = JSONGrammarConfiguration(
            tokens: Self.tokens,
            schema: .jsonObject
        ).makeMatcher()

        XCTAssertTrue(matcher.accepts(tokenID: Self.id("{")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.id("[")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.eosID))

        matcher.advance(tokenID: Self.id("{"))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("}")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.eosID))

        matcher.advance(tokenID: Self.id("}"))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id(" ")))
        XCTAssertTrue(matcher.accepts(tokenID: Self.eosID))
        XCTAssertFalse(matcher.accepts(tokenID: Self.id(",")))
    }

    func testSchemaRequiresKeysAndConstrainsTypesAndEnums() {
        let schema = JSONSchemaNode.schema(
            JSONSchema(
                allowedTypes: [.object],
                properties: [
                    "status": .schema(
                        JSONSchema(
                            allowedTypes: [.string],
                            enumValues: [.string("done"), .string("fail")]
                        )
                    )
                ],
                required: ["status"],
                additionalPropertiesAllowed: false
            )
        )
        let matcher = JSONGrammarConfiguration(tokens: Self.tokens, schema: schema).makeMatcher()

        matcher.advance(tokenID: Self.id("{"))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("\"status\"")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.id("\"age\"")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.id("}")))

        matcher.advance(tokenID: Self.id("\"status\""))
        matcher.advance(tokenID: Self.id(":"))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("\"done\"")))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("\"fail\"")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.id("\"blue\"")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.id("1")))
        XCTAssertFalse(matcher.accepts(tokenID: Self.id("true")))

        matcher.advance(tokenID: Self.id("\"done\""))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("}")))
    }

    func testTokenCanSpanStringCloseAndComma() {
        let schema = JSONSchemaNode.schema(
            JSONSchema(
                allowedTypes: [.object],
                properties: [
                    "name": .schema(JSONSchema(allowedTypes: [.string])),
                    "age": .schema(JSONSchema(allowedTypes: [.number])),
                ],
                required: ["name", "age"],
                additionalPropertiesAllowed: false
            )
        )
        let matcher = JSONGrammarConfiguration(tokens: Self.tokens, schema: schema).makeMatcher()

        for token in ["{", "\"name\"", ":"] {
            matcher.advance(tokenID: Self.id(token))
        }

        XCTAssertTrue(matcher.accepts(tokenID: Self.id("\"x\",")))
        matcher.advance(tokenID: Self.id("\"x\","))
        XCTAssertTrue(matcher.accepts(tokenID: Self.id("\"age\"")))

        matcher.advance(tokenID: Self.id("\"age\""))
        matcher.advance(tokenID: Self.id(":"))
        matcher.advance(tokenID: Self.id("1"))
        matcher.advance(tokenID: Self.id("}"))
        XCTAssertTrue(matcher.isComplete)
        XCTAssertTrue(matcher.accepts(tokenID: Self.eosID))
    }

    func testSchemaArraysAndNullsAreAcceptedWhenDeclared() {
        let schema = JSONSchemaNode.schema(
            JSONSchema(
                allowedTypes: [.object],
                properties: [
                    "items": .schema(
                        JSONSchema(
                            allowedTypes: [.array],
                            items: .schema(JSONSchema(allowedTypes: [.null]))
                        )
                    )
                ],
                required: ["items"],
                additionalPropertiesAllowed: false
            )
        )
        let matcher = JSONGrammarConfiguration(tokens: Self.tokens, schema: schema).makeMatcher()

        for token in ["{", "\"items\"", ":", "[", "null", "]", "}"] {
            XCTAssertTrue(matcher.accepts(tokenID: Self.id(token)), "expected token \(token) to be accepted")
            matcher.advance(tokenID: Self.id(token))
        }
        XCTAssertTrue(matcher.accepts(tokenID: Self.eosID))
    }

    private static let eosID = 999

    private static let tokenTable: [(String, Int)] = [
        ("{", 1),
        ("}", 2),
        ("[", 3),
        ("]", 4),
        (":", 5),
        (",", 6),
        (" ", 7),
        ("\"name\"", 8),
        ("\"age\"", 9),
        ("\"status\"", 10),
        ("\"items\"", 11),
        ("\"done\"", 12),
        ("\"fail\"", 13),
        ("\"blue\"", 14),
        ("\"x\",", 15),
        ("1", 16),
        ("true", 17),
        ("null", 18),
    ]

    private static let tokens = tokenTable.map { text, id in
        JSONGrammarToken(id: id, text: text)
    } + [
        JSONGrammarToken(id: eosID, text: "", isEOS: true)
    ]

    private static func id(_ token: String) -> Int {
        guard let match = tokenTable.first(where: { $0.0 == token }) else {
            XCTFail("unknown token \(token)")
            return -1
        }
        return match.1
    }
}
