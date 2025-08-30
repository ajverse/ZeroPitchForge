import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

class MetricsAnalyzer:
    def __init__(self, log_file="app_metrics.jsonl"):
        self.log_file = Path(log_file)
    
    def load_logs(self, days=7):
        """Load logs from the last N days"""
        if not self.log_file.exists():
            return []
        
        cutoff = datetime.now() - timedelta(days=days)
        logs = []
        
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    log = json.loads(line.strip())
                    log_time = datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00'))
                    if log_time >= cutoff:
                        logs.append(log)
                except:
                    continue
        return logs
    
    def analyze_costs(self, days=7):
        """Analyze cost patterns"""
        logs = self.load_logs(days)
        bedrock_calls = [log for log in logs if log.get('event_type') == 'bedrock_call']
        
        if not bedrock_calls:
            return {"total_tokens": 0, "estimated_cost": 0, "calls": 0}
        
        df = pd.DataFrame(bedrock_calls)
        total_tokens = df['tokens'].sum()
        estimated_cost = total_tokens * 0.000015  # $15 per 1M tokens
        
        return {
            "total_tokens": int(total_tokens),
            "estimated_cost": round(estimated_cost, 4),
            "calls": len(bedrock_calls),
            "avg_tokens_per_call": round(df['tokens'].mean(), 1),
            "avg_latency": round(df['latency_seconds'].mean(), 2)
        }
    
    def analyze_usage(self, days=7):
        """Analyze feature usage patterns"""
        logs = self.load_logs(days)
        user_actions = [log for log in logs if log.get('event_type') == 'user_action']
        
        if not user_actions:
            return {}
        
        df = pd.DataFrame(user_actions)
        usage = df['action'].value_counts().to_dict()
        
        return {
            "feature_usage": usage,
            "total_actions": len(user_actions),
            "unique_sessions": len(set(log.get('timestamp', '')[:10] for log in user_actions))
        }
    
    def generate_report(self, days=7):
        """Generate comprehensive usage report"""
        cost_analysis = self.analyze_costs(days)
        usage_analysis = self.analyze_usage(days)
        
        report = f"""
# ZeroPitchForge Usage Report (Last {days} days)

## 💰 Cost Analysis
- **Total API Calls**: {cost_analysis['calls']}
- **Total Tokens**: {cost_analysis['total_tokens']:,}
- **Estimated Cost**: ${cost_analysis['estimated_cost']}
- **Avg Tokens/Call**: {cost_analysis['avg_tokens_per_call']}
- **Avg Latency**: {cost_analysis['avg_latency']}s

## 📊 Feature Usage
- **Total Actions**: {usage_analysis.get('total_actions', 0)}
- **Active Days**: {usage_analysis.get('unique_sessions', 0)}

### Feature Breakdown:
"""
        for feature, count in usage_analysis.get('feature_usage', {}).items():
            report += f"- **{feature.replace('_', ' ').title()}**: {count}\n"
        
        return report

if __name__ == "__main__":
    analyzer = MetricsAnalyzer()
    print(analyzer.generate_report())