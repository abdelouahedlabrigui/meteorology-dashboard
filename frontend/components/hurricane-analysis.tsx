"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Loader2 } from "lucide-react"
import ReactMarkdown from 'react-markdown';


const API_BASE = "http://10.42.0.243:8003"

interface HurricaneResult {
  metadata: {
    max_hurricane: string
    east_us_lat: string
    east_us_lon: string
    west_africa_lat: string
    west_africa_lon: string
    month: string
  }
  regional_comparison: any
  us_detailed: any
  africa_detailed: any
  visualization: {
    precipitation?: { plot_encoding: string; plot_description: string }
    dispersion?: { plot_encoding: string; plot_description: string }
    temporal_intensity?: { plot_encoding: string; plot_description: string }
    enso_phase?: { plot_encoding: string; plot_description: string }
  }
  results?: string
  explanation?: string
}

export function HurricaneAnalysis() {
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<HurricaneResult[] | null>(null)
  const [formData, setFormData] = useState({
    max_hurricane: "1",
    east_us_lat: "25.7791, 32.0749",
    east_us_lon: "-80.1978, -81.0883",
    west_africa_lat: "11.8632, 14.7167",
    west_africa_lon: "15.5843, 17.4677",
    month: "september",
  })
  const [getFormData, setGetFormData] = useState({
    lat: "",
    lon: "",
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE}/weather/hurricane`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      })
      const data = await response.json()
      console.log("[v0] Hurricane POST response:", data)
      setResults([data])
    } catch (error) {
      console.error("[v0] Error fetching hurricane analysis:", error)
    } finally {
      setLoading(false)
    }
  }

  const handleGetSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE}/search/hurricane?lat=${getFormData.lat}&lon=${getFormData.lon}`)
      const data = await response.json()
      console.log("[v0] Hurricane GET response:", data)
      setResults(data.results || [data])
    } catch (error) {
      console.error("[v0] Error searching hurricanes:", error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <Tabs defaultValue="post" className="w-full">
        <TabsList className="grid w-full max-w-md grid-cols-2">
          <TabsTrigger value="post">Create Analysis</TabsTrigger>
          <TabsTrigger value="get">Search by Location</TabsTrigger>
        </TabsList>

        <TabsContent value="post" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Create Hurricane Analysis</CardTitle>
              <CardDescription>Compare hurricane patterns between US East Coast and West Africa</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="max_hurricane">Max Hurricane</Label>
                    <Input
                      id="max_hurricane"
                      value={formData.max_hurricane}
                      onChange={(e) => setFormData({ ...formData, max_hurricane: e.target.value })}
                      placeholder="1"
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="month">Month</Label>
                    <Input
                      id="month"
                      value={formData.month}
                      onChange={(e) => setFormData({ ...formData, month: e.target.value })}
                      placeholder="september"
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="east_us_lat">East US Latitude (comma-separated)</Label>
                    <Input
                      id="east_us_lat"
                      value={formData.east_us_lat}
                      onChange={(e) => setFormData({ ...formData, east_us_lat: e.target.value })}
                      placeholder="25.7791, 32.0749"
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="east_us_lon">East US Longitude (comma-separated)</Label>
                    <Input
                      id="east_us_lon"
                      value={formData.east_us_lon}
                      onChange={(e) => setFormData({ ...formData, east_us_lon: e.target.value })}
                      placeholder="-80.1978, -81.0883"
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="west_africa_lat">West Africa Latitude (comma-separated)</Label>
                    <Input
                      id="west_africa_lat"
                      value={formData.west_africa_lat}
                      onChange={(e) => setFormData({ ...formData, west_africa_lat: e.target.value })}
                      placeholder="11.8632, 14.7167"
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="west_africa_lon">West Africa Longitude (comma-separated)</Label>
                    <Input
                      id="west_africa_lon"
                      value={formData.west_africa_lon}
                      onChange={(e) => setFormData({ ...formData, west_africa_lon: e.target.value })}
                      placeholder="15.5843, 17.4677"
                      required
                    />
                  </div>
                </div>
                <Button type="submit" disabled={loading} className="w-full">
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    "Analyze Hurricanes"
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="get" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Search Hurricane Analyses</CardTitle>
              <CardDescription>Search for hurricane analyses by latitude and longitude</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleGetSearch} className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="lat">Latitude</Label>
                    <Input
                      id="lat"
                      value={getFormData.lat}
                      onChange={(e) => setGetFormData({ ...getFormData, lat: e.target.value })}
                      placeholder="13.0"
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="lon">Longitude</Label>
                    <Input
                      id="lon"
                      value={getFormData.lon}
                      onChange={(e) => setGetFormData({ ...getFormData, lon: e.target.value })}
                      placeholder="16.5"
                      required
                    />
                  </div>
                </div>
                <Button type="submit" disabled={loading} className="w-full">
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Searching...
                    </>
                  ) : (
                    "Search Analyses"
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {results && results.length > 0 && (
        <div className="space-y-6">
          {results.map((result, idx) => (
            <div key={idx} className="space-y-6">
              {/* ... existing metadata card ... */}

              {result.regional_comparison && (
                <Card>
                  <CardHeader>
                    <CardTitle>Regional Comparison</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* ... existing US and Africa analysis sections ... */}

                      {/* Add comparison section */}
                      {result.regional_comparison.comparison && (
                        <div className="col-span-full mt-4 pt-4 border-t">
                          <h4 className="font-semibold text-lg mb-2">Comparison Summary</h4>
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                              <p className="text-muted-foreground">Lambda Ratio</p>
                              <p className="font-medium">{result.regional_comparison.comparison.lambda_ratio.toFixed(2)}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">US More Active</p>
                              <p className="font-medium">{result.regional_comparison.comparison.us_more_active ? "Yes" : "No"}</p>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Add US Detailed Analysis */}
              {result.us_detailed && (
                <Card>
                  <CardHeader>
                    <CardTitle>US Detailed Analysis</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-6">
                      {result.us_detailed.negative_binomial && (
                        <div className="space-y-2">
                          <h4 className="font-semibold">Negative Binomial Model</h4>
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                            <div>
                              <p className="text-muted-foreground">r</p>
                              <p className="font-medium">{result.us_detailed.negative_binomial.r.toFixed(4)}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">p</p>
                              <p className="font-medium">{result.us_detailed.negative_binomial.p.toFixed(4)}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Overdispersion</p>
                              <p className="font-medium">{result.us_detailed.negative_binomial.overdispersion ? "Yes" : "No"}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Dispersion Ratio</p>
                              <p className="font-medium">{result.us_detailed.negative_binomial.dispersion_ratio.toFixed(2)}</p>
                            </div>
                          </div>
                        </div>
                      )}

                      {result.us_detailed.nonhomogeneous_poisson && (
                        <div className="space-y-2">
                          <h4 className="font-semibold">Non-homogeneous Poisson Process</h4>
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                            <div>
                              <p className="text-muted-foreground">Early λ</p>
                              <p className="font-medium">{result.us_detailed.nonhomogeneous_poisson.lambda_early_period.toFixed(6)}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Late λ</p>
                              <p className="font-medium">{result.us_detailed.nonhomogeneous_poisson.lambda_late_period.toFixed(6)}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Intensity Ratio</p>
                              <p className="font-medium">{result.us_detailed.nonhomogeneous_poisson.intensity_ratio.toFixed(4)}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Temporal Variability</p>
                              <p className="font-medium capitalize">{result.us_detailed.nonhomogeneous_poisson.temporal_variability}</p>
                            </div>
                          </div>
                        </div>
                      )}

                      {result.us_detailed.rainy_season && (
                        <div className="space-y-2">
                          <h4 className="font-semibold">Rainy Season Analysis</h4>
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                            <div>
                              <p className="text-muted-foreground">Rainy Days</p>
                              <p className="font-medium">{result.us_detailed.rainy_season.rainy_days}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Total Days</p>
                              <p className="font-medium">{result.us_detailed.rainy_season.total_days}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Mean Precipitation</p>
                              <p className="font-medium">{result.us_detailed.rainy_season.mean_precipitation.toFixed(4)}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Max Precipitation</p>
                              <p className="font-medium">{result.us_detailed.rainy_season.max_precipitation.toFixed(1)}</p>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Add Africa Detailed Analysis */}
              {result.africa_detailed && (
                <Card>
                  <CardHeader>
                    <CardTitle>Africa Detailed Analysis</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-6">
                      {result.africa_detailed.negative_binomial && (
                        <div className="space-y-2">
                          <h4 className="font-semibold">Negative Binomial Model</h4>
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                            <div>
                              <p className="text-muted-foreground">r</p>
                              <p className="font-medium">{result.africa_detailed.negative_binomial.r.toFixed(4)}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">p</p>
                              <p className="font-medium">{result.africa_detailed.negative_binomial.p.toFixed(4)}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Overdispersion</p>
                              <p className="font-medium">{result.africa_detailed.negative_binomial.overdispersion ? "Yes" : "No"}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Dispersion Ratio</p>
                              <p className="font-medium">{result.africa_detailed.negative_binomial.dispersion_ratio.toFixed(2)}</p>
                            </div>
                          </div>
                        </div>
                      )}

                      {result.africa_detailed.nonhomogeneous_poisson && (
                        <div className="space-y-2">
                          <h4 className="font-semibold">Non-homogeneous Poisson Process</h4>
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                            <div>
                              <p className="text-muted-foreground">Early λ</p>
                              <p className="font-medium">{result.africa_detailed.nonhomogeneous_poisson.lambda_early_period.toFixed(6)}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Late λ</p>
                              <p className="font-medium">{result.africa_detailed.nonhomogeneous_poisson.lambda_late_period.toFixed(6)}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Intensity Ratio</p>
                              <p className="font-medium">{result.africa_detailed.nonhomogeneous_poisson.intensity_ratio.toFixed(4)}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Temporal Variability</p>
                              <p className="font-medium capitalize">{result.africa_detailed.nonhomogeneous_poisson.temporal_variability}</p>
                            </div>
                          </div>
                        </div>
                      )}

                      {result.africa_detailed.rainy_season && (
                        <div className="space-y-2">
                          <h4 className="font-semibold">Rainy Season Analysis</h4>
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                            <div>
                              <p className="text-muted-foreground">Rainy Days</p>
                              <p className="font-medium">{result.africa_detailed.rainy_season.rainy_days}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Total Days</p>
                              <p className="font-medium">{result.africa_detailed.rainy_season.total_days}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Mean Precipitation</p>
                              <p className="font-medium">{result.africa_detailed.rainy_season.mean_precipitation.toFixed(4)}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Max Precipitation</p>
                              <p className="font-medium">{result.africa_detailed.rainy_season.max_precipitation.toFixed(1)}</p>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}
              {result.visualization && (
                <Card>
                  <CardHeader>
                    <CardTitle>Visualizations</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {result.visualization.precipitation && (
                        <div className="space-y-3">
                          <div className="flex justify-between items-start">
                            <h4 className="font-semibold">Precipitation Analysis</h4>
                            {result.visualization.precipitation.plot_encoding && (
                              <span className="text-xs bg-secondary px-2 py-1 rounded-full">
                                Plot Available
                              </span>
                            )}
                          </div>
                          {result.visualization.precipitation.plot_encoding ? (
                            <>
                              <img
                                src={`data:image/png;base64,${result.visualization.precipitation.plot_encoding}`}
                                alt="Precipitation Analysis"
                                className="w-full rounded-lg border border-border"
                              />
                              <div className="p-3 bg-secondary/50 rounded-lg">
                                <h5 className="font-medium text-sm mb-1">Analysis:</h5>
                                <div className="text-sm prose prose-sm max-w-none dark:prose-invert">
                                  <ReactMarkdown >
                                    {result.visualization.precipitation.plot_description}
                                  </ReactMarkdown>
                                </div>
                              </div>
                            </>
                          ) : (
                            <div className="p-4 bg-secondary/20 rounded-lg text-sm">
                              <p className="text-muted-foreground">
                                {result.visualization.precipitation.plot_description ||
                                "No precipitation plot data available"}
                              </p>
                            </div>
                          )}
                        </div>
                      )}

                      {result.visualization.dispersion && (
                        <div className="space-y-3">
                          <div className="flex justify-between items-start">
                            <h4 className="font-semibold">Dispersion Analysis</h4>
                            {result.visualization.dispersion.plot_encoding && (
                              <span className="text-xs bg-secondary px-2 py-1 rounded-full">
                                Plot Available
                              </span>
                            )}
                          </div>
                          {result.visualization.dispersion.plot_encoding ? (
                            <>
                              <img
                                src={`data:image/png;base64,${result.visualization.dispersion.plot_encoding}`}
                                alt="Dispersion Analysis"
                                className="w-full rounded-lg border border-border"
                              />
                              <div className="p-3 bg-secondary/50 rounded-lg">
                                <h5 className="font-medium text-sm mb-1">Analysis:</h5>
                                <div className="text-sm prose prose-sm max-w-none dark:prose-invert">
                                  <ReactMarkdown >
                                    {result.visualization.dispersion.plot_description}
                                  </ReactMarkdown>
                                </div>
                              </div>
                            </>
                          ) : (
                            <div className="p-4 bg-secondary/20 rounded-lg text-sm">
                              <p className="text-muted-foreground">
                                {result.visualization.dispersion.plot_description ||
                                "No dispersion plot data available"}
                              </p>
                            </div>
                          )}
                        </div>
                      )}

                      {result.visualization.temporal_intensity && (
                        <div className="space-y-3">
                          <div className="flex justify-between items-start">
                            <h4 className="font-semibold">Temporal Intensity</h4>
                            {result.visualization.temporal_intensity.plot_encoding && (
                              <span className="text-xs bg-secondary px-2 py-1 rounded-full">
                                Plot Available
                              </span>
                            )}
                          </div>
                          {result.visualization.temporal_intensity.plot_encoding ? (
                            <>
                              <img
                                src={`data:image/png;base64,${result.visualization.temporal_intensity.plot_encoding}`}
                                alt="Temporal Intensity"
                                className="w-full rounded-lg border border-border"
                              />
                              <div className="p-3 bg-secondary/50 rounded-lg">
                                <h5 className="font-medium text-sm mb-1">Analysis:</h5>
                                <div className="text-sm prose prose-sm max-w-none dark:prose-invert">
                                  <ReactMarkdown >
                                    {result.visualization.temporal_intensity.plot_description}
                                  </ReactMarkdown>
                                </div>
                              </div>
                            </>
                          ) : (
                            <div className="p-4 bg-secondary/20 rounded-lg text-sm">
                              <p className="text-muted-foreground">
                                {result.visualization.temporal_intensity.plot_description ||
                                "No temporal intensity plot data available"}
                              </p>
                            </div>
                          )}
                        </div>
                      )}

                      {result.visualization.enso_phase && (
                        <div className="space-y-3">
                          <div className="flex justify-between items-start">
                            <h4 className="font-semibold">ENSO Phase</h4>
                            {result.visualization.enso_phase.plot_encoding && (
                              <span className="text-xs bg-secondary px-2 py-1 rounded-full">
                                Plot Available
                              </span>
                            )}
                          </div>
                          {result.visualization.enso_phase.plot_encoding ? (
                            <>
                              <img
                                src={`data:image/png;base64,${result.visualization.enso_phase.plot_encoding}`}
                                alt="ENSO Phase"
                                className="w-full rounded-lg border border-border"
                              />
                              <div className="p-3 bg-secondary/50 rounded-lg">
                                <h5 className="font-medium text-sm mb-1">Analysis:</h5>
                                <div className="text-sm prose prose-sm max-w-none dark:prose-invert">
                                  <ReactMarkdown >
                                    {result.visualization.enso_phase.plot_description}
                                  </ReactMarkdown>
                                </div>
                              </div>
                            </>
                          ) : (
                            <div className="p-4 bg-secondary/20 rounded-lg text-sm">
                              <p className="text-muted-foreground">
                                {result.visualization.enso_phase.plot_description ||
                                "No ENSO phase plot data available"}
                              </p>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Update the Expert Analysis card to handle both results and explanation */}
              {(result.results || result.explanation) && (
                <Card>
                  <CardHeader>
                    <CardTitle>Expert Analysis</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {result.results && (
                      <div className="mb-4">
                        <h4 className="font-semibold mb-2">Results</h4>
                        <div className="text-sm prose prose-sm max-w-none">
                          <ReactMarkdown >
                            {result.results}
                          </ReactMarkdown>
                        </div>
                      </div>
                    )}
                    {result.explanation && (
                      <div>
                        <h4 className="font-semibold mb-2">Explanation</h4>
                        <div className="text-sm prose prose-sm max-w-none">
                          <ReactMarkdown>
                            {result.explanation}
                          </ReactMarkdown>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
