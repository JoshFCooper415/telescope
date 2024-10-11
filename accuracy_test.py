from torch import tensor
import matplotlib.pyplot as plt
# scores = [tensor([3.1538]), tensor([2.9367]), tensor([3.2513]), tensor([3.1549]), tensor([11.0467]), tensor([2.7770]), tensor([3.2411]), tensor([3.9420]), tensor([2.7951]), tensor([3.7432]), tensor([3.0816]), tensor([3.4448]), tensor([3.1055]), tensor([4.5465]), tensor([5.4150]), tensor([3.2648]), tensor([3.4142]), tensor([10.7822]), tensor([6.6682]), tensor([2.4658]), tensor([3.4590]), tensor([2.4618]), tensor([4.9129]), tensor([3.4066]), tensor([4.6021]), tensor([3.1536]), tensor([5.6573]), tensor([2.9206]), tensor([2.9717]), tensor([2.0928]), tensor([5.0104]), tensor([11.2434]), tensor([2.5615]), tensor([4.3404]), tensor([4.6811]), tensor([10.6517]), tensor([3.0735]), tensor([3.1299]), tensor([4.8839])]
# y = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

scores = [tensor([3.1538]), tensor([2.9367]), tensor([3.2513]), tensor([3.1549]), tensor([11.0467]), tensor([2.7770]), tensor([3.2411]), tensor([3.9420]), tensor([2.7951]), tensor([3.7432]), tensor([3.0816]), tensor([3.4448]), tensor([3.1055]), tensor([4.5465]), tensor([5.4150]), tensor([3.2648]), tensor([3.4142]), tensor([10.7822]), tensor([6.6682]), tensor([2.4658]), tensor([3.4590]), tensor([2.4618]), tensor([4.9129]), tensor([3.4066]), tensor([4.6021]), tensor([3.1536]), tensor([5.6573]), tensor([2.9206]), tensor([2.9717]), tensor([2.0928]), tensor([5.0104]), tensor([11.2434]), tensor([2.5615]), tensor([4.3404]), tensor([4.6811]), tensor([10.6517]), tensor([3.0735]), tensor([3.1299]), tensor([4.8839]), tensor([3.5570]), tensor([7.4805]), tensor([6.3900]), tensor([4.5716]), tensor([2.4932]), tensor([13.0964]), tensor([10.3443]), tensor([17.3527]), tensor([11.1151]), tensor([7.3778]), tensor([5.9526]), tensor([3.2292]), tensor([1.9012]), tensor([2.4952]), tensor([4.1507]), tensor([12.3709]), tensor([3.0811]), tensor([3.9796]), tensor([3.6556]), tensor([3.4051]), tensor([8.5672]), tensor([11.1846]), tensor([6.8793]), tensor([8.0031]), tensor([6.9249]), tensor([8.4572]), tensor([3.2961]), tensor([4.1558]), tensor([4.6234]), tensor([4.6123]), tensor([4.0350]), tensor([3.9019]), tensor([3.7937]), tensor([7.1587]), tensor([4.5970]), tensor([2.0219]), tensor([3.3171]), tensor([11.6243]), tensor([3.0335]), tensor([8.9751]), tensor([2.0744]), tensor([8.8452]), tensor([2.7774]), tensor([10.4637]), tensor([3.5441]), tensor([11.0477]), tensor([5.5824]), tensor([9.0073]), tensor([3.1962]), tensor([5.6195]), tensor([3.0048]), tensor([2.6169]), tensor([8.6402]), tensor([3.9251]), tensor([2.1585]), tensor([8.4096]), tensor([3.8667]), tensor([2.3945]), tensor([2.7821]), tensor([2.1680]), tensor([3.8829]), tensor([4.8681]), tensor([2.9491]), tensor([10.8823]), tensor([3.2444]), tensor([12.9574]), tensor([2.3936]), tensor([4.8852]), tensor([2.8786]), tensor([3.7873]), tensor([4.0330]), tensor([3.4220]), tensor([3.3100]), tensor([4.4643]), tensor([6.1323]), tensor([2.8511]), tensor([4.3431]), tensor([2.4926]), tensor([10.2121]), tensor([6.0098]), tensor([9.7280]), tensor([4.9470]), tensor([2.8331]), tensor([3.2318]), tensor([10.7636]), tensor([2.5670]), tensor([3.2860]), tensor([2.8547]), tensor([3.3164]), tensor([4.2082]), tensor([3.2527])]
y = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
lengths = [6156, 4079, 2695, 3684, 3870, 1549, 2402, 457, 3147, 5265, 1071, 1484, 1256, 4498, 5048, 2269, 1989, 2371, 2355, 1586, 1410, 2880, 1441, 1358, 1376, 2051, 1643, 943, 2303, 2587, 1127, 3594, 1590, 1478, 3417, 2930, 1147, 2250, 5731, 1422, 1783, 1801, 2742, 2542, 2614, 3435, 3410, 3184, 4273, 5204, 3288, 1532, 2140, 2607, 2233, 1350, 2231, 1882, 1496, 2323, 3536, 1436, 2794, 2434, 1794, 1762, 4230, 3812, 1888, 898, 2240, 2961, 1917, 2972, 1281, 1784, 3045, 2478, 1952, 5429, 1062, 1947, 2352, 1669, 2421, 989, 825, 2255, 1445, 1408, 1596, 1888, 2558, 2122, 1886, 1041, 1448, 2919, 1794, 1669, 2734, 1072, 1897, 3120, 2579, 2949, 2295, 2913, 3301, 1839, 1462, 1901, 2594, 1146, 3203, 1833, 2369, 1627, 2400, 2804, 1428, 1612, 2282, 1555, 1615, 3544, 1081, 1266, 2365, 1990]

def get_accuracy_for_length_cutoff(length_cutoff):
    number_correct = 0
    total = 0
    for i in range(len(scores)):
        if lengths[i] < length_cutoff: continue
        total += 1
        # if y_true[i] and y_scores[i] > 5.3: number_correct+=1
        # if not y_true[i] and y_scores[i] < 5.3: number_correct+=1
        
        if y[i] and scores[i] > 5.5: number_correct+=1
        if not y[i] and scores[i] < 5.5: number_correct+=1
        
    print(f"accuracy: {(number_correct)/(total)}")
    return (number_correct)/(total)
    
    
    
length_cutoffs = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000]

accuracies = []
for length_cutoff in length_cutoffs:
    accuracies.append(get_accuracy_for_length_cutoff(length_cutoff))

plt.plot(length_cutoffs, accuracies)
plt.xlabel("Length Cutoffs in Character Count")
plt.ylabel("Accuracies with 5.5 Decision Boundary")
plt.savefig("length_cutoffs_accuracies")