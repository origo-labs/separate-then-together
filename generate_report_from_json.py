"""Generate a comprehensive report from a saved session JSON."""

import json
from pathlib import Path
from separate_then_together import Config, LLMAgent, Persona, ReportGenerator

def main():
    # 1. Load Session Data
    input_path = Path("output/software_engineering_plan.json")
    if not input_path.exists():
        print(f"❌ Error: {input_path} not found. Please run examples/software_engineering.py first.")
        return

    print(f"Loading session data from {input_path}...", flush=True)
    with open(input_path, "r") as f:
        data = json.load(f)
    
    topic = data.get("topic", "Unknown Topic")
    history = data.get("results", [])
    summary_cache = data.get("summary_cache", {})
    
    # 2. Setup Generator
    config = Config.from_env()
    # Dummy persona (only used for initialization)
    agent = LLMAgent(Persona("Reporter", "Reporter"), config, initial_summary_cache=summary_cache)
    generator = ReportGenerator(agent, config)
    
    # 3. Generate Report
    print(f"\nGeneratin report for topic: {topic}", flush=True)
    print(f"History has {len(history)} turns.", flush=True)
    
    report_content = generator.generate_report(topic, history, metadata=data)
    
    # 4. Save
    output_path = Path("output/software_engineering_plan_report.md")
    with open(output_path, "w") as f:
        f.write(report_content)
        
    print(f"\n✓ Comprehensive plan saved to: {output_path}")

if __name__ == "__main__":
    main()
