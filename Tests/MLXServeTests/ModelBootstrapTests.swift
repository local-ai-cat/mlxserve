import XCTest

final class ModelBootstrapTests: XCTestCase {
    func testLocalModelResolutionSkipsCleanlyWhenAbsent() throws {
        guard let resolution = TestModelResolver.resolve() else {
            throw XCTSkip(
                "No pinned local MLX test model found. Set MLXSERVE_TEST_MODEL to run model-dependent baselines."
            )
        }

        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: resolution.url.appendingPathComponent("config.json").path
            ),
            "resolved model from \(resolution.source) must contain config.json"
        )
    }
}
