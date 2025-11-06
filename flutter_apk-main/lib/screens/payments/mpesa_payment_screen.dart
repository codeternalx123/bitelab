import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../models/mpesa_models.dart';
import '../../services/mpesa_service.dart';

class MpesaPaymentScreen extends StatefulWidget {
  final double amount;
  final String currency;
  final String subscriptionId;

  const MpesaPaymentScreen({
    super.key,
    required this.amount,
    required this.currency,
    required this.subscriptionId,
  });

  @override
  _MpesaPaymentScreenState createState() => _MpesaPaymentScreenState();
}

class _MpesaPaymentScreenState extends State<MpesaPaymentScreen> {
  final _formKey = GlobalKey<FormState>();
  final _phoneController = TextEditingController();
  bool _isProcessing = false;
  String? _error;
  Timer? _statusCheckTimer;
  String? _checkoutRequestId;

  @override
  void dispose() {
    _phoneController.dispose();
    _statusCheckTimer?.cancel();
    super.dispose();
  }

  String? _validatePhoneNumber(String? value) {
    if (value == null || value.isEmpty) {
      return 'Phone number is required';
    }
    // Assuming Kenyan phone numbers for this example
    if (!RegExp(r'^\+?254[17]\d{8}$').hasMatch(value)) {
      return 'Enter a valid Kenyan phone number';
    }
    return null;
  }

  Future<void> _initiatePayment() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _isProcessing = true;
      _error = null;
    });

    try {
      final mpesaService = Provider.of<MpesaService>(context, listen: false);
      final request = MpesaPaymentRequest(
        phoneNumber: _phoneController.text,
        amount: widget.amount,
        accountReference: widget.subscriptionId,
        description: 'Payment for subscription ${widget.subscriptionId}',
      );

      final result = await mpesaService.initiatePayment(request);

      if (!mounted) return;

      result.when(
        success: (response) {
          _checkoutRequestId = response.checkoutRequestId;
          _startStatusCheck();
          
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('STK push sent. Please check your phone.'),
              duration: Duration(seconds: 5),
            ),
          );
        },
        failure: (error) {
          setState(() {
            _error = 'Payment initiation failed: ${error.message}';
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

  void _startStatusCheck() {
    _statusCheckTimer = Timer.periodic(
      const Duration(seconds: 5),
      (timer) => _checkPaymentStatus(),
    );
  }

  Future<void> _checkPaymentStatus() async {
    if (_checkoutRequestId == null) return;

    final mpesaService = Provider.of<MpesaService>(context, listen: false);
    final result = await mpesaService.checkPaymentStatus(_checkoutRequestId!);

    result.when(
      success: (paymentResult) {
        if (paymentResult.resultCode == '0') {
          _statusCheckTimer?.cancel();
          if (mounted) {
            Navigator.of(context).pop(paymentResult); // Return success result
          }
        } else if (paymentResult.resultCode != 'pending') {
          _statusCheckTimer?.cancel();
          if (mounted) {
            setState(() {
              _error = paymentResult.resultDescription;
              _isProcessing = false;
            });
          }
        }
      },
      failure: (error) {
        // On error, we continue checking - might be temporary
        if (mounted) {
          setState(() => _error = error.message);
        }
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('M-Pesa Payment'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Amount: ${widget.currency} ${widget.amount.toStringAsFixed(2)}',
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                      const SizedBox(height: 16),
                      TextFormField(
                        controller: _phoneController,
                        decoration: const InputDecoration(
                          labelText: 'Phone Number',
                          hintText: 'Enter M-Pesa phone number',
                          prefixText: '+254 ',
                        ),
                        keyboardType: TextInputType.phone,
                        validator: _validatePhoneNumber,
                        enabled: !_isProcessing,
                      ),
                    ],
                  ),
                ),
              ),
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
                onPressed: _isProcessing ? null : _initiatePayment,
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.all(16),
                ),
                child: _isProcessing
                    ? const Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          CircularProgressIndicator(),
                          SizedBox(width: 16),
                          Text('Processing...'),
                        ],
                      )
                    : const Text('Pay with M-Pesa'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}