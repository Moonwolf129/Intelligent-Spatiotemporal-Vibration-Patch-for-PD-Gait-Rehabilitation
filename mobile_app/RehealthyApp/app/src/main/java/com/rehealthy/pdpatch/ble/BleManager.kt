package com.rehealthy.pdpatch.ble

import android.annotation.SuppressLint
import android.bluetooth.*
import android.bluetooth.le.*
import android.content.Context
import android.os.ParcelUuid
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.util.*

class BleManager(private val context: Context) {

    companion object {
        private const val DEVICE_NAME = "ReHealthyDevice"
        // Replace with your actual service/characteristic UUIDs
        val SERVICE_UUID: UUID = UUID.fromString("0000aaaa-0000-1000-8000-00805f9b34fb")
        val CHAR_IMU_UUID: UUID = UUID.fromString("0000aa01-0000-1000-8000-00805f9b34fb")
        val CHAR_GAIT_UUID: UUID = UUID.fromString("0000aa02-0000-1000-8000-00805f9b34fb")
        val CHAR_COMMAND_UUID: UUID = UUID.fromString("0000aa03-0000-1000-8000-00805f9b34fb")
    }

    private val bluetoothManager: BluetoothManager =
        context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
    private val bluetoothAdapter: BluetoothAdapter? = bluetoothManager.adapter
    private var bluetoothGatt: BluetoothGatt? = null

    private val _isConnected = MutableStateFlow(false)
    val isConnected: StateFlow<Boolean> = _isConnected

    private val _imuData = MutableStateFlow<FloatArray?>(null)
    val imuData: StateFlow<FloatArray?> = _imuData

    private val _gaitMetrics = MutableStateFlow<FloatArray?>(null)
    val gaitMetrics: StateFlow<FloatArray?> = _gaitMetrics

    private val scanCallback = object : ScanCallback() {
        override fun onScanResult(callbackType: Int, result: ScanResult) {
            val device = result.device
            if (device.name == DEVICE_NAME) {
                stopScan()
                connectDevice(device)
            }
        }
    }

    @SuppressLint("MissingPermission")
    fun startScan() {
        val scanner = bluetoothAdapter?.bluetoothLeScanner ?: return
        val settings = ScanSettings.Builder()
            .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
            .build()
        val filters = listOf(
            ScanFilter.Builder()
                .setDeviceName(DEVICE_NAME)
                .build()
        )
        scanner.startScan(filters, settings, scanCallback)
    }

    @SuppressLint("MissingPermission")
    fun stopScan() {
        val scanner = bluetoothAdapter?.bluetoothLeScanner ?: return
        scanner.stopScan(scanCallback)
    }

    @SuppressLint("MissingPermission")
    private fun connectDevice(device: BluetoothDevice) {
        bluetoothGatt = device.connectGatt(context, false, gattCallback)
    }

    private val gattCallback = object : BluetoothGattCallback() {

        override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
            if (newState == BluetoothProfile.STATE_CONNECTED) {
                _isConnected.value = true
                gatt.discoverServices()
            } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                _isConnected.value = false
            }
        }

        @SuppressLint("MissingPermission")
        override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
            val service = gatt.getService(SERVICE_UUID) ?: return
            val imuChar = service.getCharacteristic(CHAR_IMU_UUID)
            val gaitChar = service.getCharacteristic(CHAR_GAIT_UUID)

            gatt.setCharacteristicNotification(imuChar, true)
            gatt.setCharacteristicNotification(gaitChar, true)

            imuChar?.descriptors?.forEach { desc ->
                desc.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
                gatt.writeDescriptor(desc)
            }
            gaitChar?.descriptors?.forEach { desc ->
                desc.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
                gatt.writeDescriptor(desc)
            }
        }

        override fun onCharacteristicChanged(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic
        ) {
            when (characteristic.uuid) {
                CHAR_IMU_UUID -> {
                    val data = characteristic.value
                    // TODO: parse IMU data according to your firmware format
                    _imuData.value = FloatArray(7)
                }
                CHAR_GAIT_UUID -> {
                    val data = characteristic.value
                    // TODO: parse gait metrics according to your firmware format
                    _gaitMetrics.value = FloatArray(4)
                }
            }
        }
    }

    @SuppressLint("MissingPermission")
    fun sendVibrationCommand(payload: ByteArray) {
        val gatt = bluetoothGatt ?: return
        val service = gatt.getService(SERVICE_UUID) ?: return
        val cmdChar = service.getCharacteristic(CHAR_COMMAND_UUID) ?: return
        cmdChar.value = payload
        gatt.writeCharacteristic(cmdChar)
    }
}
