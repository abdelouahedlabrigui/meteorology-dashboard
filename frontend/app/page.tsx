import Link from "next/link"
import { Cloud, Database, Activity, BarChart3, ImageIcon, TrendingUp, Layers, Wind, GitBranch } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export default function Home() {
  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center gap-3">
            <Cloud className="h-8 w-8 text-primary" />
            <div>
              <h1 className="text-3xl font-bold text-foreground">MeteoAnalytics</h1>
              <p className="text-sm text-muted-foreground">Weather Intelligence Platform</p>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-12">
        <div className="mb-12 text-center">
          <h2 className="text-4xl font-bold text-foreground mb-4">Meteorological Data Analysis</h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Access comprehensive weather data, statistical analysis, and visualizations from multiple endpoints
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <Link href="/zips" className="group">
            <Card className="h-full transition-all hover:shadow-lg hover:border-primary">
              <CardHeader>
                <div className="flex items-center gap-3 mb-2">
                  <div className="p-2 rounded-lg bg-primary/10">
                    <Database className="h-6 w-6 text-primary" />
                  </div>
                  <CardTitle className="text-foreground group-hover:text-primary transition-colors">
                    ZIP Code Lookup
                  </CardTitle>
                </div>
                <CardDescription>Search and browse US ZIP codes by city, state, and county</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">Get latitude and longitude coordinates for any location</p>
              </CardContent>
            </Card>
          </Link>

          <Link href="/weather/information" className="group">
            <Card className="h-full transition-all hover:shadow-lg hover:border-primary">
              <CardHeader>
                <div className="flex items-center gap-3 mb-2">
                  <div className="p-2 rounded-lg bg-primary/10">
                    <Activity className="h-6 w-6 text-primary" />
                  </div>
                  <CardTitle className="text-foreground group-hover:text-primary transition-colors">
                    Weather Parameters
                  </CardTitle>
                </div>
                <CardDescription>View specific weather parameter data and statistics</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Temperature, humidity, wind speed, precipitation, pressure, cloud cover
                </p>
              </CardContent>
            </Card>
          </Link>

          <Link href="/weather/generated-analysis" className="group">
            <Card className="h-full transition-all hover:shadow-lg hover:border-primary">
              <CardHeader>
                <div className="flex items-center gap-3 mb-2">
                  <div className="p-2 rounded-lg bg-primary/10">
                    <TrendingUp className="h-6 w-6 text-primary" />
                  </div>
                  <CardTitle className="text-foreground group-hover:text-primary transition-colors">
                    Generated Analysis
                  </CardTitle>
                </div>
                <CardDescription>Comprehensive AI-powered weather analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">ML predictions, trends, and detailed insights</p>
              </CardContent>
            </Card>
          </Link>

          <Link href="/weather/states-analysis" className="group">
            <Card className="h-full transition-all hover:shadow-lg hover:border-primary">
              <CardHeader>
                <div className="flex items-center gap-3 mb-2">
                  <div className="p-2 rounded-lg bg-primary/10">
                    <BarChart3 className="h-6 w-6 text-primary" />
                  </div>
                  <CardTitle className="text-foreground group-hover:text-primary transition-colors">
                    Statistical Analysis
                  </CardTitle>
                </div>
                <CardDescription>Detailed statistical metrics for all parameters</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Mean, median, standard deviation, correlations, and more
                </p>
              </CardContent>
            </Card>
          </Link>

          <Link href="/weather/analysis-visualization" className="group">
            <Card className="h-full transition-all hover:shadow-lg hover:border-primary">
              <CardHeader>
                <div className="flex items-center gap-3 mb-2">
                  <div className="p-2 rounded-lg bg-primary/10">
                    <ImageIcon className="h-6 w-6 text-primary" />
                  </div>
                  <CardTitle className="text-foreground group-hover:text-primary transition-colors">
                    Visualizations
                  </CardTitle>
                </div>
                <CardDescription>Interactive charts and plots for weather data</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Time series, distributions, correlations, and predictions
                </p>
              </CardContent>
            </Card>
          </Link>

          <Link href="/weather/segments" className="group">
            <Card className="h-full transition-all hover:shadow-lg hover:border-primary">
              <CardHeader>
                <div className="flex items-center gap-3 mb-2">
                  <div className="p-2 rounded-lg bg-primary/10">
                    <Layers className="h-6 w-6 text-primary" />
                  </div>
                  <CardTitle className="text-foreground group-hover:text-primary transition-colors">
                    Weather Segments
                  </CardTitle>
                </div>
                <CardDescription>Analyze weather data in time segments</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Segmented analysis with temperature distributions and statistical insights
                </p>
              </CardContent>
            </Card>
          </Link>

          <Link href="/weather/hurricanes" className="group">
            <Card className="h-full transition-all hover:shadow-lg hover:border-primary">
              <CardHeader>
                <div className="flex items-center gap-3 mb-2">
                  <div className="p-2 rounded-lg bg-primary/10">
                    <Wind className="h-6 w-6 text-primary" />
                  </div>
                  <CardTitle className="text-foreground group-hover:text-primary transition-colors">
                    Hurricane Analysis
                  </CardTitle>
                </div>
                <CardDescription>Compare hurricane patterns between regions</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Regional comparison, ENSO analysis, and statistical modeling
                </p>
              </CardContent>
            </Card>
          </Link>

          <Link href="/weather/clusters" className="group">
            <Card className="h-full transition-all hover:shadow-lg hover:border-primary">
              <CardHeader>
                <div className="flex items-center gap-3 mb-2">
                  <div className="p-2 rounded-lg bg-primary/10">
                    <GitBranch className="h-6 w-6 text-primary" />
                  </div>
                  <CardTitle className="text-foreground group-hover:text-primary transition-colors">
                    Weather Clusters
                  </CardTitle>
                </div>
                <CardDescription>Discover weather patterns through clustering</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  K-means clustering, pattern recognition, and temporal analysis
                </p>
              </CardContent>
            </Card>
          </Link>
        </div>
      </main>
    </div>
  )
}
