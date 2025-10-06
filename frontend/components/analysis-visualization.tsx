"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { ImageIcon, Home } from "lucide-react"
import Link from "next/link"

const API_BASE = "http://10.42.0.243:8003"

interface VisualizationData {
  id: number
  basic_stats: string | null
  cross_correlations: string | null
  feature_importance: string | null
  future_predictions: string | null
  variability: string | null
  latitude: number
  longitude: number
  created: string
}

export function AnalysisVisualization() {
  const [latitude, setLatitude] = useState("")
  const [longitude, setLongitude] = useState("")
  const [data, setData] = useState<VisualizationData[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFetch = async () => {
    if (!latitude || !longitude) {
      setError("Please fill in all fields")
      return
    }

    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/weather/analysis-visualization`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          latitude: Number.parseFloat(latitude),
          longitude: Number.parseFloat(longitude),
        }),
      })

      if (!response.ok) throw new Error("Failed to fetch visualizations")

      const result = await response.json()
      console.log("[v0] API Response:", result)
      setData(Array.isArray(result) ? result : [result])
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <ImageIcon className="h-6 w-6 text-primary" />
            <h1 className="text-2xl font-bold text-foreground">Analysis Visualizations</h1>
          </div>
          <Link href="/">
            <Button variant="ghost" size="sm">
              <Home className="h-4 w-4 mr-2" />
              Home
            </Button>
          </Link>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8">
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Query Location</CardTitle>
            <CardDescription>Enter coordinates to view weather analysis visualizations</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              <div className="space-y-2">
                <Label htmlFor="latitude">Latitude</Label>
                <Input
                  id="latitude"
                  type="number"
                  step="any"
                  placeholder="e.g., 25.7791"
                  value={latitude}
                  onChange={(e) => setLatitude(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="longitude">Longitude</Label>
                <Input
                  id="longitude"
                  type="number"
                  step="any"
                  placeholder="e.g., -80.1978"
                  value={longitude}
                  onChange={(e) => setLongitude(e.target.value)}
                />
              </div>
            </div>

            <Button onClick={handleFetch} disabled={loading}>
              {loading ? "Loading..." : "Fetch Visualizations"}
            </Button>

            {error && (
              <div className="mt-4 p-4 bg-destructive/10 border border-destructive rounded-md">
                <p className="text-sm text-destructive">{error}</p>
              </div>
            )}
          </CardContent>
        </Card>

        {data.length > 0 && (
          <div className="space-y-8">
            {data.map((item) => (
              <div key={item.id} className="space-y-6">
                {/* Header Info */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-medium text-muted-foreground">Location</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm font-mono">
                        {item.latitude.toFixed(4)}, {item.longitude.toFixed(4)}
                      </p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-medium text-muted-foreground">Visualization ID</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-lg font-semibold">#{item.id}</p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-medium text-muted-foreground">Created</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm">{new Date(item.created).toLocaleString()}</p>
                    </CardContent>
                  </Card>
                </div>

                {/* Visualizations Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {item.basic_stats && (
                    <Card>
                      <CardHeader>
                        <CardTitle>Basic Statistics</CardTitle>
                        <CardDescription>Statistical distribution and summary</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <img
                          src={`data:image/png;base64,${item.basic_stats}`}
                          alt="Basic statistics visualization"
                          className="w-full h-auto rounded-lg"
                        />
                      </CardContent>
                    </Card>
                  )}

                  {item.variability && (
                    <Card>
                      <CardHeader>
                        <CardTitle>Variability Analysis</CardTitle>
                        <CardDescription>Temporal variability and patterns</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <img
                          src={`data:image/png;base64,${item.variability}`}
                          alt="Variability visualization"
                          className="w-full h-auto rounded-lg"
                        />
                      </CardContent>
                    </Card>
                  )}

                  {item.cross_correlations && (
                    <Card>
                      <CardHeader>
                        <CardTitle>Cross Correlations</CardTitle>
                        <CardDescription>Parameter relationships and dependencies</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <img
                          src={`data:image/png;base64,${item.cross_correlations}`}
                          alt="Cross correlations visualization"
                          className="w-full h-auto rounded-lg"
                        />
                      </CardContent>
                    </Card>
                  )}

                  {item.feature_importance && (
                    <Card>
                      <CardHeader>
                        <CardTitle>Feature Importance</CardTitle>
                        <CardDescription>ML model feature contributions</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <img
                          src={`data:image/png;base64,${item.feature_importance}`}
                          alt="Feature importance visualization"
                          className="w-full h-auto rounded-lg"
                        />
                      </CardContent>
                    </Card>
                  )}

                  {item.future_predictions && (
                    <Card className="lg:col-span-2">
                      <CardHeader>
                        <CardTitle>Future Predictions</CardTitle>
                        <CardDescription>6-hour forecast visualization</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <img
                          src={`data:image/png;base64,${item.future_predictions}`}
                          alt="Future predictions visualization"
                          className="w-full h-auto rounded-lg"
                        />
                      </CardContent>
                    </Card>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  )
}
