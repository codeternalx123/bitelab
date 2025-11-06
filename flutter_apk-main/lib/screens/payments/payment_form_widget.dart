import 'package:flutter/material.dart';
import 'package:provider/Provider.dart';
import '../../core/services/payment_service.dart';
import '../../models/payment_models.dart';
import 'package:local_auth/local_auth.dart';
import 'payment_status_widget.dart';

class PaymentForm extends StatefulWidget {
  final double amount;
  final String currency;
  final String subscriptionId;

  const PaymentForm({
    super.key,
    required this.amount,
    required this.currency,
    required this.subscriptionId,
  });

  @override
  _PaymentFormState createState() => _PaymentFormState();
}

class _PaymentFormState extends State<PaymentForm> {
  final _formKey = GlobalKey<FormState>();
  String _selectedPaymentMethod = 'stripe';
  bool _isProcessing = false;
  String? _error;
  final LocalAuthentication _localAuth = LocalAuthentication();

  Future<bool> _authenticateUser() async {
    try {
      final canCheckBiometrics = await _localAuth.canCheckBiometrics;
      final isDeviceSupported = await _localAuth.isDeviceSupported();

      if (canCheckBiometrics && isDeviceSupported) {
        return await _localAuth.authenticate(
          localizedReason: 'Please authenticate to process payment',
          options: const AuthenticationOptions(
            stickyAuth: true,
            biometricOnly: false,
          ),
        );
      }
      return true; // If biometrics not available, proceed anyway
    } catch (e) {
      print('Biometric authentication error: $e');
      return false;
    }
  }

  Future<void> _processPayment() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _isProcessing = true;
      _error = null;
    });

    try {
      // First authenticate the user
      final authenticated = await _authenticateUser();
      if (!authenticated) {
        setState(() {
          _error = 'Authentication failed. Please try again.';
          _isProcessing = false;
        });
        return;
      }

      final paymentService = Provider.of<PaymentService>(context, listen: false);

      final paymentRequest = PaymentRequest(
        amount: widget.amount,
        currency: widget.currency,
        paymentMethodId: _selectedPaymentMethod,
        provider: _selectedPaymentMethod,
        subscriptionId: widget.subscriptionId,
      );
      final result = await paymentService.processPayment(paymentRequest);

      if (!mounted) return;

      result.when(
        success: (payment) {

          if (payment.fraudAnalysis.isFraudulent) {
            setState(() {
              _error = 'Payment flagged as potentially fraudulent. Please try a different payment method.';
              _isProcessing = false;
            });
            return;
          }

          Navigator.of(context).push(
            MaterialPageRoute(
              builder: (context) => PaymentStatusScreen(payment: payment),
            ),
          );
        },
        failure: (error) {
          setState(() {
            _error = 'Payment failed: ${error.message}';
            _isProcessing = false;
          });
        },
      );
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = 'Payment failed: ${e.toString()}';
          _isProcessing = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Form(
      key: _formKey,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const Text(
            'Select Payment Method',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 16),
          _buildPaymentMethodSelector(),
          const SizedBox(height: 24),
          _buildPaymentSummary(),
          const SizedBox(height: 24),
          if (_error != null) ...[
            Text(
              _error!,
              style: TextStyle(
                color: Theme.of(context).colorScheme.error,
              ),
            ),
            const SizedBox(height: 16),
          ],
          ElevatedButton(
            onPressed: _isProcessing ? null : _processPayment,
            style: ElevatedButton.styleFrom(
              padding: const EdgeInsets.all(16),
            ),
            child: _isProcessing
                ? const CircularProgressIndicator()
                : const Text('Process Payment'),
          ),
          if (!_isProcessing) ...[
            const SizedBox(height: 8),
            Text(
              'Payment is secured with quantum encryption',
              style: Theme.of(context).textTheme.bodySmall,
              textAlign: TextAlign.center,
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildPaymentMethodSelector() {
    return Card(
      child: Column(
        children: [
          RadioListTile<String>(
            title: const Text('Credit Card (Stripe)'),
            subtitle: const Text('Secure payment via Stripe'),
            value: 'stripe',
            groupValue: _selectedPaymentMethod,
            onChanged: (value) {
              setState(() => _selectedPaymentMethod = value!);
            },
          ),
          RadioListTile<String>(
            title: const Text('Google Pay'),
            subtitle: const Text('Fast checkout with Google Pay'),
            value: 'google_pay',
            groupValue: _selectedPaymentMethod,
            onChanged: (value) {
              setState(() => _selectedPaymentMethod = value!);
            },
          ),
          RadioListTile<String>(
            title: const Text('Apple Pay'),
            subtitle: const Text('Quick payment with Apple Pay'),
            value: 'apple_pay',
            groupValue: _selectedPaymentMethod,
            onChanged: (value) {
              setState(() => _selectedPaymentMethod = value!);
            },
          ),
          RadioListTile<String>(
            title: const Text('Cryptocurrency'),
            subtitle: const Text('Pay with major cryptocurrencies'),
            value: 'crypto',
            groupValue: _selectedPaymentMethod,
            onChanged: (value) {
              setState(() => _selectedPaymentMethod = value!);
            },
          ),
          RadioListTile<String>(
            title: const Text('M-Pesa'),
            subtitle: const Text('Pay with M-Pesa mobile money'),
            value: 'mpesa',
            groupValue: _selectedPaymentMethod,
            onChanged: (value) {
              setState(() => _selectedPaymentMethod = value!);
            },
          ),
        ],
      ),
    );
  }

  Widget _buildPaymentSummary() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Payment Summary',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Amount:'),
                Text(
                  '${widget.currency} ${widget.amount.toStringAsFixed(2)}',
                  style: const TextStyle(fontWeight: FontWeight.bold),
                ),
              ],
            ),
            const Divider(),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Total:'),
                Text(
                  '${widget.currency} ${widget.amount.toStringAsFixed(2)}',
                  style: const TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 18,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
