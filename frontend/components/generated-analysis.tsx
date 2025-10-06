"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { TrendingUp, Home, Brain, Activity, Wind, Droplets } from "lucide-react"
import Link from "next/link"

const API_BASE = "http://10.42.0.243:8003"

interface AnalysisItem {
  id: number
  latitude: number
  longitude: number
  basic_stats_sentence: string
  basic_stats_response: string
  trend_analysis_sentence: string
  trend_analysis_response: string
  variability_sentence: string
  variability_response: string
  cross_corr_sentence: string
  cross_corr_response: string
  model_perf_sentence: string
  model_perf_response: string
  feature_importance_sentence: string
  feature_importance_response: string
  future_pred_sentence: string
  future_pred_response: string
  created: string
}

export function GeneratedAnalysis() {
  const [latitude, setLatitude] = useState("")
  const [longitude, setLongitude] = useState("")
  const [data, setData] = useState<AnalysisItem[]>([])
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
      const response = await fetch(`${API_BASE}/weather/generated-analysis`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          latitude: Number.parseFloat(latitude),
          longitude: Number.parseFloat(longitude),
        }),
      })

      if (!response.ok) throw new Error("Failed to fetch generated analysis")

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
            <TrendingUp className="h-6 w-6 text-primary" />
            <h1 className="text-2xl font-bold text-foreground">Generated Analysis</h1>
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
            <CardDescription>Enter coordinates to generate comprehensive weather analysis</CardDescription>
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
              {loading ? "Generating Analysis..." : "Generate Analysis"}
            </Button>

            {error && (
              <div className="mt-4 p-4 bg-destructive/10 border border-destructive rounded-md">
                <p className="text-sm text-destructive">{error}</p>
              </div>
            )}
          </CardContent>
        </Card>

        {data.length > 0 && (
          <div className="space-y-6">
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
                      <CardTitle className="text-sm font-medium text-muted-foreground">Analysis ID</CardTitle>
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

                {/* Basic Statistics */}
                <Card>
                  <CardHeader>
                    <div className="flex items-center gap-2">
                      <Activity className="h-5 w-5 text-primary" />
                      <CardTitle>Basic Statistics</CardTitle>
                    </div>
                    <CardDescription>{item.basic_stats_sentence}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm leading-relaxed text-foreground">{item.basic_stats_response}</p>
                  </CardContent>
                </Card>

                {/* Trend Analysis */}
                <Card>
                  <CardHeader>
                    <div className="flex items-center gap-2">
                      <TrendingUp className="h-5 w-5 text-primary" />
                      <CardTitle>Trend Analysis</CardTitle>
                    </div>
                    <CardDescription>{item.trend_analysis_sentence}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm leading-relaxed text-foreground">{item.trend_analysis_response}</p>
                  </CardContent>
                </Card>

                {/* Variability */}
                <Card>
                  <CardHeader>
                    <div className="flex items-center gap-2">
                      <Wind className="h-5 w-5 text-primary" />
                      <CardTitle>Variability</CardTitle>
                    </div>
                    <CardDescription>{item.variability_sentence}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm leading-relaxed text-foreground">{item.variability_response}</p>
                  </CardContent>
                </Card>

                {/* Cross Correlations */}
                <Card>
                  <CardHeader>
                    <div className="flex items-center gap-2">
                      <Droplets className="h-5 w-5 text-primary" />
                      <CardTitle>Cross Correlations</CardTitle>
                    </div>
                    <CardDescription>{item.cross_corr_sentence}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm leading-relaxed text-foreground">{item.cross_corr_response}</p>
                  </CardContent>
                </Card>

                {/* Model Performance */}
                <Card>
                  <CardHeader>
                    <div className="flex items-center gap-2">
                      <Brain className="h-5 w-5 text-primary" />
                      <CardTitle>Model Performance</CardTitle>
                    </div>
                    <CardDescription>{item.model_perf_sentence}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm leading-relaxed text-foreground">{item.model_perf_response}</p>
                  </CardContent>
                </Card>

                {/* Feature Importance */}
                <Card>
                  <CardHeader>
                    <CardTitle>Feature Importance</CardTitle>
                    <CardDescription>{item.feature_importance_sentence}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm leading-relaxed text-foreground">{item.feature_importance_response}</p>
                  </CardContent>
                </Card>

                {/* Future Predictions */}
                <Card>
                  <CardHeader>
                    <CardTitle>Future Predictions</CardTitle>
                    <CardDescription>{item.future_pred_sentence}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm leading-relaxed text-foreground">{item.future_pred_response}</p>
                  </CardContent>
                </Card>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  )
}
