"""
Diagnostic script to inspect CrewAI APIs in this environment.

CrewAI imports Chroma internally; on some WSL setups stdlib sqlite3 is too old for Chroma.
We apply the same sqlite3 shim used by the app before importing CrewAI.
"""

from __future__ import annotations


def _apply_sqlite_fix() -> None:
    try:
        import sqlite3

        def _t(v: str) -> tuple[int, int, int]:
            parts = (v.split(".") + ["0", "0", "0"])[:3]
            return int(parts[0]), int(parts[1]), int(parts[2])

        if _t(sqlite3.sqlite_version) < (3, 35, 0):
            import sys

            import pysqlite3

            sys.modules["sqlite3"] = pysqlite3
    except Exception:
        pass


def main() -> None:
    _apply_sqlite_fix()

    import crewai  # noqa: F401

    print("crewai_import_ok")
    print(f"crewai_version={getattr(crewai, '__version__', 'unknown')}")
    for name in ["Agent", "Task", "Crew", "Process"]:
        print(f"has_{name}={hasattr(crewai, name)}")
    print(f"has_LLM={hasattr(crewai, 'LLM')}")
    if hasattr(crewai, "LLM"):
        try:
            from inspect import signature

            print(f"LLM_init_sig={signature(crewai.LLM)}")
        except Exception as e:
            print(f"LLM_sig_failed={type(e).__name__}")

    try:
        from inspect import signature

        print(f"Agent_init_sig={signature(crewai.Agent)}")
        print(f"Task_init_sig={signature(crewai.Task)}")
        print(f"Crew_init_sig={signature(crewai.Crew)}")
        print(f"Process_members={[m for m in dir(crewai.Process) if not m.startswith('_')][:20]}")
    except Exception as e:
        print(f"sig_introspection_failed={type(e).__name__}")

    try:
        import crewai.tools as ct  # type: ignore

        print("crewai_tools_import_ok")
        for name in ["BaseTool", "tool"]:
            print(f"crewai_tools_has_{name}={hasattr(ct, name)}")
    except Exception as e:
        print(f"crewai_tools_import_failed={type(e).__name__}")


if __name__ == "__main__":
    main()


