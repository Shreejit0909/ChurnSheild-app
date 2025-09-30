import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  Shield, 
  Brain, 
  BarChart3, 
  Users, 
  Zap, 
  Target,
  TrendingUp,
  FileText,
  Settings,
  Upload,
  CheckCircle,
  ArrowRight,
  Sparkles
} from "lucide-react";
import logo from "@/assets/Logo.png";
import heroImage from "@/assets/dashboard-hero-bg.jpg";
import { useNavigate } from "react-router-dom";

const features = [
  {
    icon: Brain,
    title: "AI-Powered Predictions",
    description: "Advanced machine learning algorithms predict customer churn with 92% accuracy"
  },
  {
    icon: BarChart3,
    title: "Real-time Analytics",
    description: "Interactive dashboards and charts provide deep insights into churn patterns"
  },
  {
    icon: Target,
    title: "Risk Segmentation",
    description: "Automatically categorize customers into low, medium, and high-risk segments"
  },
  {
    icon: Upload,
    title: "Batch Processing",
    description: "Upload CSV files to predict churn for thousands of customers at once"
  },
  {
    icon: Users,
    title: "Customer Insights",
    description: "Understand which factors contribute most to customer churn"
  },
  {
    icon: Zap,
    title: "Real-time Alerts",
    description: "Get notified instantly when high-risk customers are detected"
  }
];

const steps = [
  {
    number: "01",
    title: "Upload Customer Data",
    description: "Import your customer data via CSV or connect to your existing database"
  },
  {
    number: "02", 
    title: "Configure Model",
    description: "Set up prediction parameters and risk thresholds for your business"
  },
  {
    number: "03",
    title: "Generate Predictions",
    description: "Run AI predictions and get actionable insights about customer churn risk"
  },
  {
    number: "04",
    title: "Take Action",
    description: "Use insights to create targeted retention campaigns and reduce churn"
  }
];

export default function GetStarted() {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate('/dashboard');
  };

  return (
    <div className="min-h-screen bg-gradient-subtle">
      {/* Header */}
      <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <img src={logo} alt="ChurnShield AI" className="w-10 h-10" />
              <div>
                <h1 className="text-xl font-bold bg-gradient-primary bg-clip-text text-transparent">
                  ChurnShield AI
                </h1>
                <p className="text-xs text-muted-foreground">Customer Churn Prediction</p>
              </div>
            </div>
            <Button onClick={handleGetStarted} className="bg-gradient-primary">
              Go to Dashboard
              <ArrowRight className="w-4 h-4 ml-2" />
            </Button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section 
        className="relative py-20 px-4 text-center overflow-hidden"
        style={{
          backgroundImage: `linear-gradient(135deg, hsl(var(--brand-primary) / 0.95), hsl(var(--brand-secondary) / 0.95)), url(${heroImage})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
        }}
      >
        <div className="container mx-auto relative z-10">
          <Badge className="mb-6 bg-white/20 text-white border-white/30">
            <Sparkles className="w-4 h-4 mr-2" />
            AI-Powered Customer Retention
          </Badge>
          
          <h1 className="text-5xl md:text-6xl font-bold text-white mb-6">
            Predict & Prevent
            <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-white to-white/80">
              Customer Churn
            </span>
          </h1>
          
          <p className="text-xl text-white/90 mb-8 max-w-2xl mx-auto">
            Use advanced AI to identify at-risk customers before they leave. 
            Reduce churn by up to 35% with predictive analytics and actionable insights.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button 
              size="lg" 
              onClick={handleGetStarted}
              className="bg-white text-brand-primary hover:bg-white/90 shadow-large"
            >
              Get Started Free
              <ArrowRight className="w-5 h-5 ml-2" />
            </Button>
            <Button 
              size="lg" 
              variant="outline" 
              className="border-white/30 text-white hover:bg-white/10"
            >
              Watch Demo
            </Button>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-20 px-4">
        <div className="container mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Powerful Features</h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Everything you need to understand, predict, and prevent customer churn
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature) => (
              <Card key={feature.title} className="bg-gradient-card shadow-soft hover:shadow-medium transition-shadow">
                <CardHeader>
                  <div className="w-12 h-12 rounded-lg bg-gradient-primary flex items-center justify-center mb-4">
                    <feature.icon className="w-6 h-6 text-white" />
                  </div>
                  <CardTitle className="text-xl">{feature.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">{feature.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-20 px-4 bg-muted/30">
        <div className="container mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">How It Works</h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Get up and running with ChurnShield AI in just four simple steps
            </p>
          </div>
          
          <div className="grid lg:grid-cols-4 gap-8">
            {steps.map((step, index) => (
              <div key={step.number} className="text-center">
                <div className="relative mb-6">
                  <div className="w-16 h-16 mx-auto rounded-full bg-gradient-primary flex items-center justify-center text-white font-bold text-xl shadow-brand">
                    {step.number}
                  </div>
                  {index < steps.length - 1 && (
                    <div className="hidden lg:block absolute top-8 left-full w-full h-0.5 bg-gradient-to-r from-brand-primary to-brand-secondary"></div>
                  )}
                </div>
                <h3 className="text-xl font-semibold mb-3">{step.title}</h3>
                <p className="text-muted-foreground">{step.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20 px-4">
        <div className="container mx-auto">
          <div className="grid md:grid-cols-3 gap-8 text-center">
            <div className="space-y-2">
              <div className="text-4xl font-bold text-brand-primary">92%</div>
              <div className="text-lg font-semibold">Prediction Accuracy</div>
              <div className="text-muted-foreground">Industry-leading ML algorithms</div>
            </div>
            <div className="space-y-2">
              <div className="text-4xl font-bold text-brand-primary">35%</div>
              <div className="text-lg font-semibold">Churn Reduction</div>
              <div className="text-muted-foreground">Average customer improvement</div>
            </div>
            <div className="space-y-2">
              <div className="text-4xl font-bold text-brand-primary">10k+</div>
              <div className="text-lg font-semibold">Customers Analyzed</div>
              <div className="text-muted-foreground">Trusted by growing businesses</div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 bg-gradient-primary text-white">
        <div className="container mx-auto text-center">
          <h2 className="text-4xl font-bold mb-4">Ready to Reduce Churn?</h2>
          <p className="text-xl text-white/90 mb-8 max-w-2xl mx-auto">
            Join thousands of businesses using ChurnShield AI to retain customers and increase revenue.
          </p>
          <Button 
            size="lg" 
            onClick={handleGetStarted}
            className="bg-white text-brand-primary hover:bg-white/90 shadow-large"
          >
            Start Your Free Trial
            <ArrowRight className="w-5 h-5 ml-2" />
          </Button>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-4 border-t">
        <div className="container mx-auto text-center">
          <div className="flex items-center justify-center gap-3 mb-4">
            <img src={logo} alt="ChurnShield AI" className="w-8 h-8" />
            <span className="text-lg font-bold bg-gradient-primary bg-clip-text text-transparent">
              ChurnShield AI
            </span>
          </div>
          <p className="text-muted-foreground">
            © 2024 ChurnShield AI. All rights reserved. | Advanced Customer Churn Prediction Platform
          </p>
        </div>
      </footer>
    </div>
  );
}