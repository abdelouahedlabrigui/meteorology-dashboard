import { HurricaneAnalysis } from "@/components/hurricane-analysis"

export default function HurricanesPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-6 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">Hurricane Analysis</h1>
          <p className="text-muted-foreground">
            Compare hurricane patterns between US East Coast and West Africa regions
          </p>
        </div>
        <HurricaneAnalysis />
      </div>
    </div>
  )
}
