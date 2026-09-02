import benchmarkData from "@/data/benchmark.json";
import demoCasesData from "@/data/demo_cases.json";
import projectStatusData from "@/data/project_status.json";
import { PortfolioDemo } from "@/components/portfolio-demo";
import type { BenchmarkSnapshot, DemoCase, ProjectStatus } from "@/lib/types";

export default function Home() {
  return (
    <PortfolioDemo
      benchmark={benchmarkData as BenchmarkSnapshot}
      cases={demoCasesData as DemoCase[]}
      projectStatus={projectStatusData as ProjectStatus}
    />
  );
}
