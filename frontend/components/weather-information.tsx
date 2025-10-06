"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Activity, Home, TrendingUp, TrendingDown, Brain } from "lucide-react"
import Link from "next/link"

const API_BASE = "http://10.42.0.243:8003"

const PARAMETERS = [
  { value: "temperature", label: "Temperature" },
  { value: "humidity", label: "Humidity" },
  { value: "wind_speed", label: "Wind Speed" },
  { value: "precipitation", label: "Precipitation" },
  { value: "pressure", label: "Pressure" },
  { value: "cloud_cover", label: "Cloud Cover" },
]

interface WeatherData {
  statistical_analysis: {
    basic_stats: {
      sentence: string
      text_generation: string
      data: {
        mean: number
        median: number
        std: number
        min: number
        max: number
        range: number
      }
    }
    trend_analysis: {
      sentence: string
      text_generation: string
      data: {
        correlation_with_time: number
        is_increasing: boolean
      }
    }
    variability: {
      sentence: string
      text_generation: string
      data: {
        coefficient_of_variation: number
        percentiles: {
          "25th": number
          "75th": number
          "90th": number
        }
      }
    }
    cross_correlations: {
      sentence: string
      text_generation: string
      data: {
        temperature?: number
        humidity?: number
        wind_speed?: number
      }
    }
  }
  numerical_predictions: {
    model_performance: {
      sentence: string
      text_generation: string
      data: {
        mean_absolute_error: number
        root_mean_squared_error: number
        prediction_accuracy: number
      }
    }
    feature_importance: {
      sentence: string
      text_generation: string
      data: Record<string, number>
    }
    future_predictions: {
      sentence: string
      text_generation: string
      data: {
        next_6_hours: number[]
        timestamps: string[]
      }
    }
  }
  temperature_analysis: {
    parameter: string
    llm_analysis: string
    visualization?: string
  }
}

export function WeatherInformation() {
  const [latitude, setLatitude] = useState("")
  const [longitude, setLongitude] = useState("")
  const [parameter, setParameter] = useState("")
  const [data, setData] = useState<WeatherData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFetch = async () => {
    if (!latitude || !longitude || !parameter) {
      setError("Please fill in all fields")
      return
    }

    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/weather/information`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          latitude: Number.parseFloat(latitude),
          longitude: Number.parseFloat(longitude),
          parameter,
        }),
      })

      if (!response.ok) throw new Error("Failed to fetch weather information")

      const result = await response.json()
      console.log("[v0] API Response:", result)
      setData(result)
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
            <Activity className="h-6 w-6 text-primary" />
            <h1 className="text-2xl font-bold text-foreground">Weather Parameters</h1>
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
            <CardTitle>Query Weather Data</CardTitle>
            <CardDescription>Enter coordinates and select a parameter to analyze</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
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
              <div className="space-y-2">
                <Label htmlFor="parameter">Parameter</Label>
                <Select value={parameter} onValueChange={setParameter}>
                  <SelectTrigger id="parameter">
                    <SelectValue placeholder="Select parameter" />
                  </SelectTrigger>
                  <SelectContent>
                    {PARAMETERS.map((param) => (
                      <SelectItem key={param.value} value={param.value}>
                        {param.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <Button onClick={handleFetch} disabled={loading}>
              {loading ? "Loading..." : "Fetch Data"}
            </Button>

            {error && (
              <div className="mt-4 p-4 bg-destructive/10 border border-destructive rounded-md">
                <p className="text-sm text-destructive">{error}</p>
              </div>
            )}
          </CardContent>
        </Card>

        {data && (
          <div className="space-y-6">
            {/* LLM Analysis Card */}
            {data.temperature_analysis?.llm_analysis && (
              <Card>
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <Brain className="h-5 w-5 text-primary" />
                    <CardTitle>AI Analysis - {data.temperature_analysis.parameter}</CardTitle>
                  </div>
                  <CardDescription>Advanced GPT-2 generated insights</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-sm leading-relaxed text-foreground">{data.temperature_analysis.llm_analysis}</p>
                </CardContent>
              </Card>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Basic Statistics */}
              <Card>
                <CardHeader>
                  <CardTitle>Basic Statistics</CardTitle>
                  <CardDescription>{data.statistical_analysis.basic_stats.sentence}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Mean:</span>
                      <span className="font-medium">{data.statistical_analysis.basic_stats.data.mean.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Median:</span>
                      <span className="font-medium">
                        {data.statistical_analysis.basic_stats.data.median.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Std Dev:</span>
                      <span className="font-medium">{data.statistical_analysis.basic_stats.data.std.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Min:</span>
                      <span className="font-medium">{data.statistical_analysis.basic_stats.data.min.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Max:</span>
                      <span className="font-medium">{data.statistical_analysis.basic_stats.data.max.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Range:</span>
                      <span className="font-medium">{data.statistical_analysis.basic_stats.data.range.toFixed(2)}</span>
                    </div>
                  </div>
                  <div className="mt-4 pt-4 border-t border-border">
                    <p className="text-xs text-muted-foreground leading-relaxed">
                      {data.statistical_analysis.basic_stats.text_generation}
                    </p>
                  </div>
                </CardContent>
              </Card>

              {/* Trend Analysis */}
              <Card>
                <CardHeader>
                  <CardTitle>Trend Analysis</CardTitle>
                  <CardDescription>{data.statistical_analysis.trend_analysis.sentence}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        {data.statistical_analysis.trend_analysis.data.is_increasing ? (
                          <TrendingUp className="h-5 w-5 text-green-500" />
                        ) : (
                          <TrendingDown className="h-5 w-5 text-red-500" />
                        )}
                        <span className="font-medium">
                          {data.statistical_analysis.trend_analysis.data.is_increasing ? "Increasing" : "Decreasing"}
                        </span>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Correlation with time:{" "}
                        {data.statistical_analysis.trend_analysis.data.correlation_with_time.toFixed(4)}
                      </p>
                    </div>
                    <div className="pt-4 border-t border-border">
                      <p className="text-xs text-muted-foreground leading-relaxed">
                        {data.statistical_analysis.trend_analysis.text_generation}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Variability */}
              <Card>
                <CardHeader>
                  <CardTitle>Variability</CardTitle>
                  <CardDescription>{data.statistical_analysis.variability.sentence}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Coefficient of Variation:</span>
                      <span className="font-medium">
                        {data.statistical_analysis.variability.data.coefficient_of_variation.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">25th Percentile:</span>
                      <span className="font-medium">
                        {data.statistical_analysis.variability.data.percentiles["25th"].toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">75th Percentile:</span>
                      <span className="font-medium">
                        {data.statistical_analysis.variability.data.percentiles["75th"].toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">90th Percentile:</span>
                      <span className="font-medium">
                        {data.statistical_analysis.variability.data.percentiles["90th"].toFixed(2)}
                      </span>
                    </div>
                  </div>
                  <div className="mt-4 pt-4 border-t border-border">
                    <p className="text-xs text-muted-foreground leading-relaxed">
                      {data.statistical_analysis.variability.text_generation}
                    </p>
                  </div>
                </CardContent>
              </Card>

              {/* Cross Correlations */}
              <Card>
                <CardHeader>
                  <CardTitle>Cross Correlations</CardTitle>
                  <CardDescription>{data.statistical_analysis.cross_correlations.sentence}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {Object.entries(data.statistical_analysis.cross_correlations.data).map(([key, value]) => (
                      <div key={key} className="flex justify-between">
                        <span className="text-muted-foreground capitalize">{key.replace("_", " ")}:</span>
                        <span className="font-medium">{value?.toFixed(4) ?? "N/A"}</span>
                      </div>
                    ))}
                  </div>
                  <div className="mt-4 pt-4 border-t border-border">
                    <p className="text-xs text-muted-foreground leading-relaxed">
                      {data.statistical_analysis.cross_correlations.text_generation}
                    </p>
                  </div>
                </CardContent>
              </Card>

              {/* Model Performance */}
              <Card>
                <CardHeader>
                  <CardTitle>Model Performance</CardTitle>
                  <CardDescription>{data.numerical_predictions.model_performance.sentence}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">MAE:</span>
                      <span className="font-medium">
                        {data.numerical_predictions.model_performance.data.mean_absolute_error.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">RMSE:</span>
                      <span className="font-medium">
                        {data.numerical_predictions.model_performance.data.root_mean_squared_error.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Accuracy:</span>
                      <span className="font-medium">
                        {(data.numerical_predictions.model_performance.data.prediction_accuracy * 100).toFixed(2)}%
                      </span>
                    </div>
                  </div>
                  <div className="mt-4 pt-4 border-t border-border">
                    <p className="text-xs text-muted-foreground leading-relaxed">
                      {data.numerical_predictions.model_performance.text_generation}
                    </p>
                  </div>
                </CardContent>
              </Card>

              {/* Feature Importance */}
              <Card>
                <CardHeader>
                  <CardTitle>Feature Importance</CardTitle>
                  <CardDescription>{data.numerical_predictions.feature_importance.sentence}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {Object.entries(data.numerical_predictions.feature_importance.data)
                      .sort(([, a], [, b]) => b - a)
                      .map(([key, value]) => (
                        <div key={key} className="space-y-1">
                          <div className="flex justify-between text-sm">
                            <span className="text-muted-foreground capitalize">{key.replace("_", " ")}</span>
                            <span className="font-medium">{value.toFixed(4)}</span>
                          </div>
                          <div className="h-2 bg-muted rounded-full overflow-hidden">
                            <div className="h-full bg-primary" style={{ width: `${value * 100}%` }} />
                          </div>
                        </div>
                      ))}
                  </div>
                  <div className="mt-4 pt-4 border-t border-border">
                    <p className="text-xs text-muted-foreground leading-relaxed">
                      {data.numerical_predictions.feature_importance.text_generation}
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Future Predictions */}
            <Card>
              <CardHeader>
                <CardTitle>6-Hour Forecast</CardTitle>
                <CardDescription>{data.numerical_predictions.future_predictions.sentence}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                  {data.numerical_predictions.future_predictions.data.next_6_hours.map((value, index) => {
                    const timestamp = new Date(data.numerical_predictions.future_predictions.data.timestamps[index])
                    return (
                      <div key={index} className="text-center p-4 bg-muted rounded-lg">
                        <p className="text-xs text-muted-foreground mb-1">
                          {timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                        </p>
                        <p className="text-lg font-semibold">{value.toFixed(2)}</p>
                      </div>
                    )
                  })}
                </div>
                <div className="mt-4 pt-4 border-t border-border">
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    {data.numerical_predictions.future_predictions.text_generation}
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Visualization */}
            {data.temperature_analysis?.visualization && (
              <Card>
                <CardHeader>
                  <CardTitle>Visualization</CardTitle>
                  <CardDescription>Statistical analysis plot</CardDescription>
                </CardHeader>
                <CardContent>
                  <img
                    src={`data:image/png;base64,${data.temperature_analysis.visualization}`}
                    alt="Weather analysis visualization"
                    className="w-full h-auto rounded-lg"
                  />
                </CardContent>
              </Card>
            )}
          </div>
        )}
      </main>
    </div>
  )
}
