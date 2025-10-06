import { WeatherClusters } from "@/components/weather-clusters"

export default function ClustersPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-6 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">Weather Clusters Analysis</h1>
          <p className="text-muted-foreground">
            Discover weather patterns through K-means clustering and temporal analysis
          </p>
        </div>
        <WeatherClusters />
      </div>
    </div>
  )
}
