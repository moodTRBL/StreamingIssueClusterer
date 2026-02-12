from __future__ import annotations

import logging
from orchestrator import PipelineOrchestrator


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = PipelineOrchestrator(config_path="resources/config.yml").run()
    print(f"scraped items: {result.scraped}")
    print(f"saved items: {result.saved}")
    print(f"clustered items: {result.clustered}")

    print("issues:")
    for issue in result.issues:
        print(
            f"- id={issue.id} title={issue.title} "
            f"article_count={issue.article_count} updated_at={issue.updated_at.isoformat()}"
        )


if __name__ == "__main__":
    main()
