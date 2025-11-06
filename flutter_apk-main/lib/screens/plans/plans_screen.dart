import 'package:flutter/material.dart';
import '../../services/api_service.dart';

class PlansScreen extends StatefulWidget {
  const PlansScreen({super.key});

  @override
  State<PlansScreen> createState() => _PlansScreenState();
}

class _PlansScreenState extends State<PlansScreen> {
  final ApiService _api = ApiService();
  List plans = [];
  bool loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final resp = await _api.get('/api/v1/plans');
    setState(() { plans = resp.data ?? []; loading = false; });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Plans')),
      body: loading ? const Center(child: CircularProgressIndicator()) : ListView.builder(itemCount: plans.length, itemBuilder: (_,i){
        final p = plans[i];
        return ListTile(title: Text(p['name'] ?? 'Plan'), subtitle: Text(p['description'] ?? ''), trailing: ElevatedButton(onPressed: (){}, child: const Text('Subscribe')));
      }),
    );
  }
}
