import 'package:flutter/material.dart';
import '../../services/api_service.dart';
import 'package:url_launcher/url_launcher.dart';

class ReportsScreen extends StatefulWidget {
  const ReportsScreen({super.key});

  @override
  State<ReportsScreen> createState() => _ReportsScreenState();
}

class _ReportsScreenState extends State<ReportsScreen> {
  final ApiService _api = ApiService();
  List reports = [];
  bool loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final resp = await _api.get('/api/v1/reports');
    setState(() { reports = resp.data ?? []; loading = false; });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Reports')),
      body: loading ? const Center(child: CircularProgressIndicator()) : ListView.builder(itemCount: reports.length, itemBuilder: (_,i){
        final r = reports[i];
        final urlString = r['report_url'] ?? '';
        final url = Uri.parse(urlString);
        return ListTile(title: Text(r['title'] ?? 'Report'), trailing: IconButton(icon: const Icon(Icons.open_in_new), onPressed: () async { if (await canLaunchUrl(url)) await launchUrl(url); }));
      }),
    );
  }
}
