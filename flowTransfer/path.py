import net


distances = [10, 50, 100, 200, 500]

def getM(distances):
    wireless = net.WirelessMetrics(frequency_ghz=2.4, tx_power_dbm=20, antenna_gain_db=2)

    # 测试不同距离的性能指标
    print("距离(m) | 能耗(mW) | 延迟(ms) | 带宽(Mbps) | 丢包率(%)")
    print("-" * 60)
    cost = []
    delaySet = []
    band = []
    ratio = []

    for distance in distances:
        power = wireless.calculate_power_consumption(distance)
        delay = wireless.calculate_delay(distance)
        bandwidth = wireless.calculate_bandwidth(distance)
        packet_loss = wireless.calculate_packet_loss(distance)

        cost.append(power)
        delaySet.append(delay)
        band.append(bandwidth)
        ratio.append(packet_loss)

        print(f"{distance:7.1f} | {power:8.2f} | {delay:8.3f} | {bandwidth:9.2f} | {packet_loss:9.2f}")

    return cost, delaySet, band, ratio

a,b,c,d = getM(distances)
print(a,b,c,d)