"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Cloud, Database, Activity, TrendingUp, BarChart3, ImageIcon, Layers, Wind, GitBranch } from "lucide-react"
import { cn } from "@/lib/utils"

const navItems = [
  { href: "/", label: "Home", icon: Cloud },
  { href: "/zips", label: "ZIP Codes", icon: Database },
  { href: "/weather/information", label: "Parameters", icon: Activity },
  { href: "/weather/generated-analysis", label: "AI Analysis", icon: TrendingUp },
  { href: "/weather/states-analysis", label: "Statistics", icon: BarChart3 },
  { href: "/weather/analysis-visualization", label: "Visualizations", icon: ImageIcon },
  { href: "/weather/segments", label: "Segments", icon: Layers },
  { href: "/weather/hurricanes", label: "Hurricanes", icon: Wind },
  { href: "/weather/clusters", label: "Clusters", icon: GitBranch },
]

export function Navbar() {
  const pathname = usePathname()

  return (
    <nav className="border-b border-border bg-card sticky top-0 z-50">
      <div className="container mx-auto px-6">
        <div className="flex items-center justify-between py-4">
          <Link href="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
            <Cloud className="h-7 w-7 text-primary" />
            <div>
              <h1 className="text-xl font-bold text-foreground">MeteoAnalytics</h1>
              <p className="text-xs text-muted-foreground">Weather Intelligence</p>
            </div>
          </Link>

          <div className="flex items-center gap-1 overflow-x-auto">
            {navItems.slice(1).map((item) => {
              const Icon = item.icon
              const isActive = pathname === item.href
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors whitespace-nowrap",
                    isActive
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground hover:bg-accent",
                  )}
                >
                  <Icon className="h-4 w-4" />
                  {item.label}
                </Link>
              )
            })}
          </div>
        </div>
      </div>
    </nav>
  )
}
