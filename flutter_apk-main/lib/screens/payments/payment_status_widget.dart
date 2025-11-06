import 'package:flutter/material.dart';
import '../../models/payment_models.dart';

class PaymentStatusScreen extends StatelessWidget {
  final Payment payment;

  const PaymentStatusScreen({
    super.key,
    required this.payment,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Payment Status'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildStatusCard(),
            const SizedBox(height: 16),
            _buildPaymentDetailsCard(),
            const SizedBox(height: 16),
            _buildSecurityAnalysisCard(),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  payment.status == 'success'
                      ? Icons.check_circle
                      : Icons.error,
                  color: payment.status == 'success'
                      ? Colors.green
                      : Colors.red,
                  size: 48,
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        payment.status == 'success'
                            ? 'Payment Successful'
                            : 'Payment Failed',
                        style: const TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Transaction ID: ${payment.id}',
                        style: const TextStyle(color: Colors.grey),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPaymentDetailsCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Payment Details',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            _buildDetailRow('Amount', '${payment.currency} ${payment.amount}'),
            _buildDetailRow('Provider', payment.provider),
            _buildDetailRow(
              'Date',
              DateTime.parse(payment.createdAt).toLocal().toString(),
            ),
            _buildDetailRow('Provider ID', payment.providerPaymentId),
          ],
        ),
      ),
    );
  }

  Widget _buildSecurityAnalysisCard() {
    final analysis = payment.fraudAnalysis;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Security Analysis',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            _buildDetailRow(
              'Risk Level',
              analysis.riskLevel,
              color: _getRiskColor(analysis.riskLevel),
            ),
            _buildDetailRow(
              'Fraud Probability',
              '${(analysis.fraudProbability * 100).toStringAsFixed(1)}%',
            ),
            const SizedBox(height: 16),
            const Text(
              'Risk Factors',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            _buildRiskFactors(analysis.featureImportance),
          ],
        ),
      ),
    );
  }

  Widget _buildDetailRow(String label, String value, {Color? color}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: const TextStyle(color: Colors.grey),
          ),
          Text(
            value,
            style: TextStyle(
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRiskFactors(FeatureImportance features) {
    return Column(
      children: [
        _buildRiskFactor('Amount', features.amount),
        _buildRiskFactor('Frequency', features.frequency),
        _buildRiskFactor('Time Pattern', features.timePattern),
        _buildRiskFactor('Location Risk', features.locationRisk),
        _buildRiskFactor('Device Risk', features.deviceRisk),
      ],
    );
  }

  Widget _buildRiskFactor(String label, double value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Expanded(
            flex: 3,
            child: Text(label),
          ),
          Expanded(
            flex: 7,
            child: LinearProgressIndicator(
              value: value,
              backgroundColor: Colors.grey[200],
              valueColor: AlwaysStoppedAnimation<Color>(
                _getRiskColor(_getRiskLevel(value)),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Color _getRiskColor(String riskLevel) {
    switch (riskLevel.toLowerCase()) {
      case 'low':
        return Colors.green;
      case 'medium':
        return Colors.orange;
      case 'high':
        return Colors.red;
      default:
        return Colors.grey;
    }
  }

  String _getRiskLevel(double value) {
    if (value < 0.3) return 'low';
    if (value < 0.7) return 'medium';
    return 'high';
  }
}