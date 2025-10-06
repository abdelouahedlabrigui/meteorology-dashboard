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

interface ClusterResult {
  metadata?: {
    latitude: number
    longitude: number
    elevation: number
    timezone: string
    analysis_period: {
      start: string
      end: string
    }
    clustering_method: string
    number_of_clusters: number
  }
  clusters?: Record<string, {
    size: number
    statistics: any
    dominant_weather: string
    hour_distribution: Record<string, number>
    day_distribution: Record<string, number>
    description: string
  }>
  occurrences?: Record<string, string[]>
  visualizations?: Record<string, {
    plot_encoding: string
    plot_description: string
  }>
  full_report?: string
}

export function WeatherClusters() {
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<ClusterResult[] | null>(null)
  const [formData, setFormData] = useState({
    filepath:
      "/home/labrigui/Software/microservices/python-software/conversationalai/apis/weather/repos/analysis/data/meteo_1.json",
    max_winters: "1",
    latitude: "",
    longitude: "",
    month: "june",
  })
  const [getFormData, setGetFormData] = useState({
    lat: "",
    lon: "",
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE}/weather/clusters`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filepath: formData.filepath,
          max_winters: Number.parseInt(formData.max_winters),
          latitude: Number.parseFloat(formData.latitude),
          longitude: Number.parseFloat(formData.longitude),
          month: formData.month,
        }),
      })
      const data = await response.json()
      console.log("[v0] Clusters POST response:", data)
      setResults([data])
    } catch (error) {
      console.error("[v0] Error fetching cluster analysis:", error)
    } finally {
      setLoading(false)
    }
  }

  const handleGetSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE}/search/clusters?lat=${getFormData.lat}&lon=${getFormData.lon}`)
      const data = await response.json()
      console.log("[v0] Clusters GET response:", data)
      setResults(data.results || [data])
    } catch (error) {
      console.error("[v0] Error searching clusters:", error)
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
              <CardTitle>Create Cluster Analysis</CardTitle>
              <CardDescription>Discover weather patterns through K-means clustering</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2 md:col-span-2">
                    <Label htmlFor="filepath">File Path</Label>
                    <Input
                      id="filepath"
                      value={formData.filepath}
                      onChange={(e) => setFormData({ ...formData, filepath: e.target.value })}
                      placeholder="/path/to/meteo_data.json"
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="max_winters">Max Winters</Label>
                    <Input
                      id="max_winters"
                      type="number"
                      value={formData.max_winters}
                      onChange={(e) => setFormData({ ...formData, max_winters: e.target.value })}
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
                      placeholder="june"
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="latitude">Latitude</Label>
                    <Input
                      id="latitude"
                      value={formData.latitude}
                      onChange={(e) => setFormData({ ...formData, latitude: e.target.value })}
                      placeholder="38.917"
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="longitude">Longitude</Label>
                    <Input
                      id="longitude"
                      value={formData.longitude}
                      onChange={(e) => setFormData({ ...formData, longitude: e.target.value })}
                      placeholder="-119.9865"
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
                    "Analyze Clusters"
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="get" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Search Cluster Analyses</CardTitle>
              <CardDescription>Search for cluster analyses by latitude and longitude</CardDescription>
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
                      placeholder="38.910366"
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="lon">Longitude</Label>
                    <Input
                      id="lon"
                      value={getFormData.lon}
                      onChange={(e) => setGetFormData({ ...getFormData, lon: e.target.value })}
                      placeholder="-119.91792"
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
        {/* Metadata Card */}
        {result.metadata && (
          <Card>
            <CardHeader>
              <CardTitle>Analysis Metadata</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground">Location</p>
                  <p className="font-medium">{result.metadata.latitude}, {result.metadata.longitude}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Elevation</p>
                  <p className="font-medium">{result.metadata.elevation}m</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Period</p>
                  <p className="font-medium">
                    {new Date(result.metadata.analysis_period.start).toLocaleDateString()} to {
                      new Date(result.metadata.analysis_period.end).toLocaleDateString()
                    }
                  </p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Method</p>
                  <p className="font-medium">{result.metadata.clustering_method}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Clusters Card */}
        {result.clusters && (
          <Card>
            <CardHeader>
              <CardTitle>Weather Clusters</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {Object.entries(result.clusters).map(([clusterId, clusterData]: [string, any]) => (
                  <Card key={clusterId} className="bg-accent/50">
                    <CardHeader>
                      <CardTitle className="text-base">Cluster {clusterId}</CardTitle>
                      <CardDescription>{clusterData.description}</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <p className="text-muted-foreground">Size</p>
                          <p className="font-medium">{clusterData.size} hours</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Temperature</p>
                          <p className="font-medium">
                            {clusterData.statistics.temperature.mean.toFixed(1)}°C
                          </p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Humidity</p>
                          <p className="font-medium">
                            {clusterData.statistics.humidity.mean.toFixed(1)}%
                          </p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Wind Speed</p>
                          <p className="font-medium">
                            {clusterData.statistics.wind_speed.mean.toFixed(1)} km/h
                          </p>
                        </div>
                      </div>

                      {/* Hour Distribution Chart */}
                      <div className="mt-4">
                        <h4 className="font-semibold mb-2">Hourly Distribution</h4>
                        <div className="grid grid-cols-12 gap-1">
                          {Array.from({length: 24}).map((_, hour) => (
                            <div
                              key={hour}
                              className="flex flex-col items-center"
                              style={{
                                height: `${(clusterData.hour_distribution[hour] || 0) * 100}%`,
                                minHeight: '20px'
                              }}
                            >
                              <div className="w-full bg-primary rounded-t"
                                   style={{height: '100%'}}></div>
                              <span className="text-xs mt-1">{hour}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
              )}

        {/* Visualizations Card */}
        {result.visualizations && (
          <Card>
            <CardHeader>
              <CardTitle>Visualizations</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {Object.entries(result.visualizations).map(([vizName, vizData]) => (
                  <div key={vizName} className="space-y-2">
                    <h4 className="font-semibold capitalize">{vizName.replace('_', ' ')}</h4>
                    <img
                      src={`data:image/png;base64,${vizData.plot_encoding}`}
                      alt={vizName}
                      className="w-full rounded-lg border border-border"
                    />
                    <p className="text-sm text-muted-foreground">
                      <ReactMarkdown>{vizData.plot_description}</ReactMarkdown>
                    </p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Full Report */}
        {result.full_report && (
          <Card>
            <CardHeader>
              <CardTitle>Full Analysis Report</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="prose max-w-none">
                <ReactMarkdown>{result.full_report}</ReactMarkdown>
              </div>
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
