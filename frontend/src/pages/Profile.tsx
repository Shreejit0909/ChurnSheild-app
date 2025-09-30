import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Users, Zap, Rocket, Shield } from "lucide-react";
import { motion } from "framer-motion";

export default function Profile() {
  return (
    <div className="flex-1 space-y-10 p-8">
      {/* Hero Section */}
      <div className="text-center space-y-4">
        <motion.h1
          className="text-5xl font-extrabold bg-gradient-to-r from-pink-500 via-purple-500 to-indigo-500 bg-clip-text text-transparent"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          About ChurnSheild AI
        </motion.h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          ChurnSheild AI is an AI-powered partner to <span className="font-semibold">reduce churn</span>,
          <span className="font-semibold"> retain customers</span>, and
          <span className="font-semibold"> increase lifetime value</span>.
        </p>
      </div>

      {/* Mission + Vision */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="bg-gradient-to-br from-indigo-500/10 to-pink-500/10">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Rocket className="h-5 w-5 text-indigo-500" />
              Our Mission
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p>
              To empower businesses with <strong>predictive intelligence</strong>
              that identifies at-risk customers and provides <strong>actionable retention strategies</strong>.
            </p>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-pink-500/10 to-indigo-500/10">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-pink-500" />
              Our Vision
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p>
              To become the industry standard for <strong>customer retention</strong> —
              combining AI, data, and human creativity to reduce churn worldwide.
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Timeline Section */}
      <Card>
        <CardHeader>
          <CardTitle>Our Journey</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="relative pl-6 border-l-2 border-muted-foreground/20 space-y-6">
            <div className="relative">
              <span className="absolute -left-3 top-1.5 w-3 h-3 rounded-full bg-indigo-500"></span>
              <h3 className="font-semibold">2024 — Idea Born 💡</h3>
              <p className="text-muted-foreground">
                We saw companies struggling with customer churn and built an AI model to predict it.
              </p>
            </div>
            <div className="relative">
              <span className="absolute -left-3 top-1.5 w-3 h-3 rounded-full bg-pink-500"></span>
              <h3 className="font-semibold">2025 — Platform Launch 🚀</h3>
              <p className="text-muted-foreground">
                Released ChurnSheild AI with real-time churn predictions, batch processing, Retention Suggestions & analytics dashboards.
              </p>
            </div>
            
          </div>
        </CardContent>
      </Card>

      {/* Trust Section */}
      <Card>
        <CardHeader>
          <CardTitle>Why Companies Trust Us</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
          <div>
            <Users className="h-10 w-10 text-indigo-500 mx-auto" />
            <h4 className="font-semibold mt-2">10K+ Customers Analyzed</h4>
            <p className="text-sm text-muted-foreground">
              Already helping businesses understand customer behavior at scale.
            </p>
          </div>
          <div>
            <Zap className="h-10 w-10 text-pink-500 mx-auto" />
            <h4 className="font-semibold mt-2">Real-time Predictions</h4>
            <p className="text-sm text-muted-foreground">
              Get churn insights instantly and take immediate retention actions.
            </p>
          </div>
          <div>
            <Rocket className="h-10 w-10 text-purple-500 mx-auto" />
            <h4 className="font-semibold mt-2">AI + Human Strategy</h4>
            <p className="text-sm text-muted-foreground">
              Combine the power of AI with your team’s expertise for maximum impact.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Footer */}
      <footer className="text-center py-6 text-sm text-muted-foreground border-t mt-8">
        © {new Date().getFullYear()} RetainAI — <span className="italic">“Predict. Prevent. Retain.”</span>
      </footer>
    </div>
  );
}
