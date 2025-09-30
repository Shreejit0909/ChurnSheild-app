"""
Professional Synthetic Customer Data Generator
Industry-standard approach to creating realistic datasets for ML training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict
import json
from pathlib import Path


class CustomerDataGenerator:
    """
    Synthetic Customer Data Generator

    Why a class instead of functions?
    - State management: Keep configuration in one place
    - Extensibility: Easy to add new customer types
    - Testing: Can mock individual methods
    - Industry pattern: Data generators are always classes in production
    """

    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)
        self.random_seed = random_seed

        # Industry-realistic parameters based on SaaS companies
        self.customer_segments = {
            'enterprise': {'weight': 0.15, 'churn_rate': 0.05},
            'mid_market': {'weight': 0.25, 'churn_rate': 0.12},
            'small_business': {'weight': 0.45, 'churn_rate': 0.18},
            'startup': {'weight': 0.15, 'churn_rate': 0.25}
        }

    def generate_customer_base(self, n_customers: int = 10000) -> pd.DataFrame:
        customers = []

        for customer_id in range(1, n_customers + 1):
            segment = self._assign_customer_segment()
            signup_date = self._generate_signup_date()
            tenure_days = (datetime.now() - signup_date).days

            usage_metrics = self._generate_usage_patterns(segment, tenure_days)
            support_metrics = self._generate_support_metrics(segment, usage_metrics['usage_trend'])
            financial_metrics = self._generate_financial_metrics(segment)
            demographics = self._generate_demographics(segment)

            churn_probability = self._calculate_churn_probability(
                segment, tenure_days, usage_metrics, support_metrics, financial_metrics
            )
            is_churned = np.random.random() < churn_probability

            customer = {
                'customer_id': f'CUST_{customer_id:06d}',
                'signup_date': signup_date,
                'tenure_days': tenure_days,
                'segment': segment,
                'is_churned': is_churned,
                'churn_probability': round(churn_probability, 4),
                **usage_metrics,
                **support_metrics,
                **financial_metrics,
                **demographics
            }

            customers.append(customer)

        df = pd.DataFrame(customers)
        df = self._add_derived_features(df)

        print(f"✅ Generated {len(df)} customers")
        print(f"📊 Churn Rate: {df['is_churned'].mean():.2%}")
        print(f"🏢 Segment Distribution:\n{df['segment'].value_counts(normalize=True)}")

        return df

    def _assign_customer_segment(self) -> str:
        segments = list(self.customer_segments.keys())
        weights = [self.customer_segments[s]['weight'] for s in segments]
        return np.random.choice(segments, p=weights)

    def _generate_signup_date(self) -> datetime:
        days_ago = np.random.exponential(365)
        days_ago = min(days_ago, 1095)
        return datetime.now() - timedelta(days=int(days_ago))

    def _generate_usage_patterns(self, segment: str, tenure_days: int) -> Dict:
        base_usage = {
            'enterprise': np.random.normal(450, 100),
            'mid_market': np.random.normal(180, 50),
            'small_business': np.random.normal(85, 25),
            'startup': np.random.normal(35, 15)
        }[segment]

        base_usage = max(0, base_usage)

        trend_factor = np.random.choice(
            [0.8, 0.9, 1.0, 1.1, 1.2],
            p=[0.15, 0.20, 0.30, 0.25, 0.10]
        )

        current_usage = base_usage * trend_factor

        max_features = {'enterprise': 25, 'mid_market': 15, 'small_business': 8, 'startup': 5}[segment]
        features_used = np.random.poisson(max_features * 0.6)
        features_used = min(features_used, max_features)

        days_since_login = np.random.exponential(7) if trend_factor >= 1.0 else np.random.exponential(21)
        days_since_login = min(days_since_login, 90)

        return {
            'monthly_usage': round(current_usage, 1),
            'usage_trend': round(trend_factor, 2),
            'features_used': features_used,
            'days_since_last_login': round(days_since_login, 1),
            'avg_session_duration': round(np.random.exponential(45), 1)
        }

    def _generate_support_metrics(self, segment: str, usage_trend: float) -> Dict:
        base_tickets = np.random.poisson(2) if usage_trend >= 1.0 else np.random.poisson(6)

        resolution_time = {
            'enterprise': np.random.exponential(4),
            'mid_market': np.random.exponential(12),
            'small_business': np.random.exponential(24),
            'startup': np.random.exponential(48)
        }[segment]

        satisfaction = max(1, min(5, 5 - (resolution_time / 12)))

        return {
            'support_tickets_90d': base_tickets,
            'avg_resolution_time': round(resolution_time, 1),
            'support_satisfaction': round(satisfaction, 1),
            'escalated_tickets': min(base_tickets, np.random.poisson(0.5))
        }

    def _generate_financial_metrics(self, segment: str) -> Dict:
        mrr = {
            'enterprise': np.random.normal(5000, 1500),
            'mid_market': np.random.normal(800, 200),
            'small_business': np.random.normal(150, 50),
            'startup': np.random.normal(50, 20)
        }[segment]

        mrr = max(10, mrr)

        payment_failures = np.random.poisson(0.3)

        contract_type = np.random.choice(
            ['monthly', 'annual', 'multi_year'],
            p=[0.6, 0.35, 0.05] if segment in ['startup', 'small_business'] else [0.3, 0.6, 0.1]
        )

        return {
            'monthly_revenue': round(mrr, 2),
            'payment_failures_90d': payment_failures,
            'contract_type': contract_type,
            'days_to_renewal': np.random.randint(1, 365)
        }

    def _generate_demographics(self, segment: str) -> Dict:
        company_size = {
            'enterprise': np.random.randint(1000, 10000),
            'mid_market': np.random.randint(100, 1000),
            'small_business': np.random.randint(10, 100),
            'startup': np.random.randint(2, 50)
        }[segment]

        industries = ['Technology', 'Finance', 'Healthcare', 'Retail', 'Manufacturing', 'Education']
        industry = np.random.choice(industries)

        regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America']
        region = np.random.choice(regions, p=[0.5, 0.3, 0.15, 0.05])

        return {
            'company_size': company_size,
            'industry': industry,
            'region': region
        }

    def _calculate_churn_probability(self, segment: str, tenure_days: int,
                                     usage_metrics: Dict, support_metrics: Dict,
                                     financial_metrics: Dict) -> float:
        base_churn = self.customer_segments[segment]['churn_rate']

        usage_factor = 1.0
        if usage_metrics['usage_trend'] < 0.9:
            usage_factor *= 2.5
        if usage_metrics['days_since_last_login'] > 30:
            usage_factor *= 2.0
        if usage_metrics['features_used'] < 2:
            usage_factor *= 1.8

        support_factor = 1.0
        if support_metrics['support_tickets_90d'] > 5:
            support_factor *= 1.8
        if support_metrics['support_satisfaction'] < 3:
            support_factor *= 2.2

        financial_factor = 1.0
        if financial_metrics['payment_failures_90d'] > 0:
            financial_factor *= 3.0
        if financial_metrics['days_to_renewal'] < 30:
            financial_factor *= 1.5

        tenure_factor = 1.0
        if tenure_days < 90:
            tenure_factor *= 1.8
        elif tenure_days > 730:
            tenure_factor *= 0.7

        total_factor = usage_factor * support_factor * financial_factor * tenure_factor
        final_probability = base_churn * total_factor
        return min(0.95, final_probability)

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['lifecycle_stage'] = pd.cut(
            df['tenure_days'],
            bins=[0, 30, 90, 365, np.inf],
            labels=['onboarding', 'early', 'mature', 'veteran']
        )

        df['usage_intensity'] = (
            df['monthly_usage'] / df['company_size'] * 1000
        ).round(2)

        # ✅ Fixed vectorized tenure scaling
        tenure_scaled = (df['tenure_days'] / 365).clip(upper=2) / 2

        df['manual_risk_score'] = (
            (2 - df['usage_trend']) * 0.35 +
            (df['support_tickets_90d'] / 10) * 0.25 +
            (df['payment_failures_90d'] / 3) * 0.20 +
            (1 - df['features_used'] / 10) * 0.15 +
            (1 - tenure_scaled) * 0.05
        ).round(3)

        df['revenue_per_employee'] = (df['monthly_revenue'] / df['company_size']).round(2)

        return df

    def save_dataset(self, df: pd.DataFrame, output_dir: str = "data/raw") -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"customer_data_{timestamp}.csv.gz"
        filepath = Path(output_dir) / filename

        df.to_csv(filepath, index=False, compression='gzip')

        metadata = {
            'generated_at': datetime.now().isoformat(),
            'total_customers': len(df),
            'churn_rate': float(df['is_churned'].mean()),
            'features': list(df.columns),
            'segments': df['segment'].value_counts().to_dict(),
            'random_seed': self.random_seed
        }

        metadata_path = Path(output_dir) / f"metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"💾 Dataset saved: {filepath}")
        print(f"📋 Metadata saved: {metadata_path}")

        return str(filepath)


if __name__ == "__main__":
    generator = CustomerDataGenerator(random_seed=42)
    df = generator.generate_customer_base(n_customers=10000)
    filepath = generator.save_dataset(df)

    print("\n📊 Dataset Summary:")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Missing values: {df.isnull().sum().sum()}")

    print("\n🎯 Target Distribution:")
    print(df.groupby(['segment', 'is_churned']).size().unstack(fill_value=0))
