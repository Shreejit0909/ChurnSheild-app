import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

const mockCustomers = [
  {
    id: "CUST-001",
    segment: "Enterprise",
    churnProbability: 0.92,
    riskLevel: "High"
  },
  {
    id: "CUST-002", 
    segment: "SMB",
    churnProbability: 0.87,
    riskLevel: "High"
  },
  {
    id: "CUST-003",
    segment: "Enterprise", 
    churnProbability: 0.78,
    riskLevel: "High"
  },
  {
    id: "CUST-004",
    segment: "Mid-Market",
    churnProbability: 0.74,
    riskLevel: "Medium"
  },
  {
    id: "CUST-005",
    segment: "SMB",
    churnProbability: 0.69,
    riskLevel: "Medium"
  }
];

const getRiskBadgeVariant = (risk: string) => {
  switch (risk) {
    case "High": return "destructive";
    case "Medium": return "secondary";
    case "Low": return "default";
    default: return "default";
  }
};

export const HighRiskCustomers = () => {
  return (
    <Card className="bg-gradient-card shadow-soft">
      <CardHeader>
        <CardTitle className="text-lg font-semibold">High-Risk Customers</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Customer ID</TableHead>
              <TableHead>Segment</TableHead>
              <TableHead>Churn Probability</TableHead>
              <TableHead>Risk Level</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {mockCustomers.map((customer) => (
              <TableRow key={customer.id}>
                <TableCell className="font-medium">{customer.id}</TableCell>
                <TableCell>{customer.segment}</TableCell>
                <TableCell>{(customer.churnProbability * 100).toFixed(1)}%</TableCell>
                <TableCell>
                  <Badge variant={getRiskBadgeVariant(customer.riskLevel)}>
                    {customer.riskLevel}
                  </Badge>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
};