"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { BarChart3, Home, Brain, TrendingUp, TrendingDown } from "lucide-react"
import Link from "next/link"

const API_BASE = "http://10.42.0.243:8003"

interface ParameterAnalysis {
  id: number
  parameter: string
  llm_analysis: string
  mean_value: number
  median_value: number
  std_dev: number
  min_value: number
  max_value: number
  range_value: number
  correlation_with_time: number
  is_increasing: string
  coeff_of_variation: number
  percentile_25: number
  percentile_75: number
  percentile_90: number
  corr_humidity: number | null
  corr_wind_speed: number | null
  corr_pressure: number | null
  mae: number
  rmse: number
  prediction_accuracy: number
  fi_pressure: number | null
  fi_temperature: number | null
  fi_lag_1: number | null
  fi_lag_2: number | null
  fi_wind_speed: number | null
  pred_hour1: number
  pred_hour2: number
  pred_hour3: number
  pred_hour4: number
  pred_hour5: number
  pred_hour6: number
  ts_hour1: string
  ts_hour2: string
  ts_hour3: string
  ts_hour4: string
  ts_hour5: string
  ts_hour6: string
  algorithm: string
  features_used: string
  training_samples: number
  analysis_timestamp: string
  latitude: number
  longitude: number
  data_points: number
  parameter_unit: string
  methods_used: string
  visualization: string | null
  created: string
}

export function StatesAnalysis() {
  const [latitude, setLatitude] = useState("")
  const [longitude, setLongitude] = useState("")
  const [data, setData] = useState<ParameterAnalysis[]>([])
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
      const response = await fetch(`${API_BASE}/weather/states-analysis`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          latitude: Number.parseFloat(latitude),
          longitude: Number.parseFloat(longitude),
        }),
      })

      if (!response.ok) throw new Error("Failed to fetch states analysis")

      const result = await response.json()
      console.log("[v0] API Response:", result)
      setData(Array.isArray(result) ? result : [result])
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setLoading(false)
    }
  }

  const renderParameterCard = (param: ParameterAnalysis) => (
    <div className="space-y-6">
      {/* LLM Analysis */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            <CardTitle>AI Analysis</CardTitle>
          </div>
          <CardDescription>Advanced GPT-2 generated insights</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm leading-relaxed text-foreground">{param.llm_analysis}</p>
        </CardContent>
      </Card>

      {/* Statistics */}
      <Card>
        <CardHeader>
          <CardTitle>Statistical Summary</CardTitle>
          <CardDescription>
            {param.data_points} data points • {param.parameter_unit}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-muted-foreground">Mean</p>
              <p className="text-2xl font-bold">{param.mean_value.toFixed(2)}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Median</p>
              <p className="text-2xl font-bold">{param.median_value.toFixed(2)}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Std Dev</p>
              <p className="text-2xl font-bold">{param.std_dev.toFixed(2)}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Min</p>
              <p className="text-lg font-semibold">{param.min_value.toFixed(2)}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Max</p>
              <p className="text-lg font-semibold">{param.max_value.toFixed(2)}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Range</p>
              <p className="text-lg font-semibold">{param.range_value.toFixed(2)}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Trend & Variability */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Trend Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  {param.is_increasing === "Y" ? (
                    <TrendingUp className="h-5 w-5 text-green-500" />
                  ) : (
                    <TrendingDown className="h-5 w-5 text-red-500" />
                  )}
                  <span className="font-medium">{param.is_increasing === "Y" ? "Increasing" : "Decreasing"}</span>
                </div>
                <p className="text-sm text-muted-foreground">
                  Correlation with time: {param.correlation_with_time.toFixed(4)}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Variability Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-muted-foreground">CV</p>
                <p className="text-lg font-semibold">{param.coeff_of_variation.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">P25</p>
                <p className="text-lg font-semibold">{param.percentile_25.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">P75</p>
                <p className="text-lg font-semibold">{param.percentile_75.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">P90</p>
                <p className="text-lg font-semibold">{param.percentile_90.toFixed(2)}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Cross-Correlations */}
      <Card>
        <CardHeader>
          <CardTitle>Cross-Correlations</CardTitle>
          <CardDescription>Relationships with other parameters</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {param.corr_humidity !== null && (
              <div className="flex justify-between items-center">
                <span className="text-sm">Humidity</span>
                <span className="font-mono text-sm">{param.corr_humidity.toFixed(4)}</span>
              </div>
            )}
            {param.corr_wind_speed !== null && (
              <div className="flex justify-between items-center">
                <span className="text-sm">Wind Speed</span>
                <span className="font-mono text-sm">{param.corr_wind_speed.toFixed(4)}</span>
              </div>
            )}
            {param.corr_pressure !== null && (
              <div className="flex justify-between items-center">
                <span className="text-sm">Pressure</span>
                <span className="font-mono text-sm">{param.corr_pressure.toFixed(4)}</span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Model Performance */}
      <Card>
        <CardHeader>
          <CardTitle>ML Model Performance</CardTitle>
          <CardDescription>
            {param.algorithm} • {param.training_samples} training samples
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div>
              <p className="text-sm text-muted-foreground">MAE</p>
              <p className="text-xl font-bold">{param.mae.toFixed(4)}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">RMSE</p>
              <p className="text-xl font-bold">{param.rmse.toFixed(4)}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Accuracy</p>
              <p className="text-xl font-bold">{(param.prediction_accuracy * 100).toFixed(2)}%</p>
            </div>
          </div>

          <div className="pt-4 border-t border-border">
            <p className="text-sm font-medium mb-3">Feature Importance</p>
            <div className="space-y-2">
              {param.fi_pressure !== null && (
                <div className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Pressure</span>
                    <span className="font-medium">{param.fi_pressure.toFixed(4)}</span>
                  </div>
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div className="h-full bg-primary" style={{ width: `${param.fi_pressure * 100}%` }} />
                  </div>
                </div>
              )}
              {param.fi_temperature !== null && (
                <div className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Temperature</span>
                    <span className="font-medium">{param.fi_temperature.toFixed(4)}</span>
                  </div>
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div className="h-full bg-primary" style={{ width: `${param.fi_temperature * 100}%` }} />
                  </div>
                </div>
              )}
              {param.fi_wind_speed !== null && (
                <div className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Wind Speed</span>
                    <span className="font-medium">{param.fi_wind_speed.toFixed(4)}</span>
                  </div>
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div className="h-full bg-primary" style={{ width: `${param.fi_wind_speed * 100}%` }} />
                  </div>
                </div>
              )}
              {param.fi_lag_1 !== null && (
                <div className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Lag 1</span>
                    <span className="font-medium">{param.fi_lag_1.toFixed(4)}</span>
                  </div>
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div className="h-full bg-primary" style={{ width: `${param.fi_lag_1 * 100}%` }} />
                  </div>
                </div>
              )}
              {param.fi_lag_2 !== null && (
                <div className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Lag 2</span>
                    <span className="font-medium">{param.fi_lag_2.toFixed(4)}</span>
                  </div>
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div className="h-full bg-primary" style={{ width: `${param.fi_lag_2 * 100}%` }} />
                  </div>
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Future Predictions */}
      <Card>
        <CardHeader>
          <CardTitle>6-Hour Forecast</CardTitle>
          <CardDescription>Predicted values for the next 6 hours</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {[
              { value: param.pred_hour1, time: param.ts_hour1 },
              { value: param.pred_hour2, time: param.ts_hour2 },
              { value: param.pred_hour3, time: param.ts_hour3 },
              { value: param.pred_hour4, time: param.ts_hour4 },
              { value: param.pred_hour5, time: param.ts_hour5 },
              { value: param.pred_hour6, time: param.ts_hour6 },
            ].map((pred, index) => {
              const timestamp = new Date(pred.time)
              return (
                <div key={index} className="text-center p-4 bg-muted rounded-lg">
                  <p className="text-xs text-muted-foreground mb-1">
                    {timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                  </p>
                  <p className="text-lg font-semibold">{pred.value.toFixed(2)}</p>
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>

      {/* Visualization */}
      {param.visualization && (
        <Card>
          <CardHeader>
            <CardTitle>Visualization</CardTitle>
            <CardDescription>Statistical analysis plot</CardDescription>
          </CardHeader>
          <CardContent>
            <img
              src={`data:image/png;base64,${param.visualization}`}
              alt={`${param.parameter} visualization`}
              className="w-full h-auto rounded-lg"
            />
          </CardContent>
        </Card>
      )}

      {/* Metadata */}
      <Card>
        <CardHeader>
          <CardTitle>Analysis Metadata</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-muted-foreground">Analysis Timestamp</p>
              <p className="font-medium">{new Date(param.analysis_timestamp).toLocaleString()}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Methods Used</p>
              <p className="font-medium">{JSON.parse(param.methods_used).join(", ")}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Features Used</p>
              <p className="font-medium text-xs">{JSON.parse(param.features_used).join(", ")}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Created</p>
              <p className="font-medium">{new Date(param.created).toLocaleString()}</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <BarChart3 className="h-6 w-6 text-primary" />
            <h1 className="text-2xl font-bold text-foreground">Statistical Analysis</h1>
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
            <CardDescription>Enter coordinates to view detailed statistical analysis</CardDescription>
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
              {loading ? "Loading..." : "Fetch Analysis"}
            </Button>

            {error && (
              <div className="mt-4 p-4 bg-destructive/10 border border-destructive rounded-md">
                <p className="text-sm text-destructive">{error}</p>
              </div>
            )}
          </CardContent>
        </Card>

        {data.length > 0 && (
          <Tabs defaultValue={data[0].parameter} className="w-full">
            <TabsList className="grid w-full grid-cols-2 lg:grid-cols-3">
              {data.map((param) => (
                <TabsTrigger key={param.id} value={param.parameter}>
                  {param.parameter}
                </TabsTrigger>
              ))}
            </TabsList>

            {data.map((param) => (
              <TabsContent key={param.id} value={param.parameter}>
                {renderParameterCard(param)}
              </TabsContent>
            ))}
          </Tabs>
        )}
      </main>
    </div>
  )
}
