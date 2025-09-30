import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import logo from "@/assets/Logo.png";

interface SplashScreenProps {
  onComplete: () => void;
}

export const SplashScreen = ({ onComplete }: SplashScreenProps) => {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(timer);
          setTimeout(onComplete, 500); // Small delay before hiding
          return 100;
        }
        return prev + 2;
      });
    }, 30);

    return () => clearInterval(timer);
  }, [onComplete]);

  return (
    <motion.div 
      className="fixed inset-0 z-50 flex items-center justify-center bg-gradient-primary"
      initial={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="text-center space-y-8">
        <motion.div
          initial={{ scale: 0.5, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="mx-auto"
        >
          <img 
            src={logo} 
            alt="ChurnShield AI" 
            className="w-24 h-24 mx-auto mb-6"
          />
          <h1 className="text-4xl font-bold text-white mb-2">ChurnShield AI</h1>
          <p className="text-white/80 text-lg">Customer Churn Prediction Platform</p>
        </motion.div>

        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.5, duration: 0.6 }}
          className="w-64 mx-auto"
        >
          <div className="w-full bg-white/20 rounded-full h-2 mb-4">
            <motion.div 
              className="bg-white h-2 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
          <p className="text-white/70 text-sm">Loading dashboard...</p>
        </motion.div>
      </div>
    </motion.div>
  );
};