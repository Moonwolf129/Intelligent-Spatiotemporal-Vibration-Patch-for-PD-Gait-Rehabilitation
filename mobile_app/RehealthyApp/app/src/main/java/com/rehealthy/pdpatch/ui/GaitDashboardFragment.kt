package com.rehealthy.pdpatch.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.rehealthy.pdpatch.R
import com.rehealthy.pdpatch.ble.BleManager
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

class GaitDashboardFragment : Fragment() {

    private lateinit var bleManager: BleManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        bleManager = BleManager(requireContext())
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val view = inflater.inflate(R.layout.fragment_gait_dashboard, container, false)
        val btnConnect: Button = view.findViewById(R.id.btn_connect)
        val tvStatus: TextView = view.findViewById(R.id.tv_status)
        val tvGait: TextView = view.findViewById(R.id.tv_gait)

        btnConnect.setOnClickListener {
            bleManager.startScan()
        }

        viewLifecycleOwner.lifecycleScope.launch {
            bleManager.isConnected.collectLatest { connected ->
                tvStatus.text = if (connected) "Connected" else "Disconnected"
            }
        }

        viewLifecycleOwner.lifecycleScope.launch {
            bleManager.gaitMetrics.collectLatest { metrics ->
                metrics?.let {
                    tvGait.text = "Gait metrics: " + it.joinToString(", ")
                }
            }
        }

        return view
    }

    override fun onPause() {
        super.onPause()
        bleManager.stopScan()
    }

    companion object {
        fun newInstance() = GaitDashboardFragment()
    }
}
