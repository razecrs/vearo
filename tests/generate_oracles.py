import json
import numpy as np

def make_case(op, inputs, output):
    return {
        "op": op,
        "inputs": [{"shape": list(x.shape), "data": x.flatten().tolist()} for x in inputs],
        "output": {"shape": list(output.shape), "data": output.flatten().tolist()}
    }

def main():
    cases = []

    # 1. Add
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    y = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
    cases.append(make_case("add", [x, y], x + y))

    # Add Transposed
    x_t = x.T # shape [3, 2]
    y_t = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32)
    cases.append(make_case("add_transposed", [x_t, y_t], x_t + y_t))

    # Add Broadcasted
    x_b = np.array([[1.0, 2.0, 3.0]], dtype=np.float32) # shape [1, 3]
    y_b = np.array([[10.0], [20.0]], dtype=np.float32) # shape [2, 1]
    cases.append(make_case("add_broadcasted", [x_b, y_b], x_b + y_b))

    # 2. Sub
    cases.append(make_case("sub", [x, y], x - y))

    # 3. Mul
    cases.append(make_case("mul", [x, y], x * y))

    # 4. Div
    cases.append(make_case("div", [x, y], x / y))

    # 5. Matmul 2D
    m1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    m2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    cases.append(make_case("matmul_2d", [m1, m2], np.matmul(m1, m2)))

    # Matmul Batched
    mb1 = np.arange(1, 13, dtype=np.float32).reshape(2, 2, 3)
    mb2 = np.arange(1, 13, dtype=np.float32).reshape(2, 3, 2)
    cases.append(make_case("matmul_batched", [mb1, mb2], np.matmul(mb1, mb2)))

    # 6. Reshape
    r1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    cases.append(make_case("reshape", [r1], r1.reshape(6)))

    # 7. Transpose
    t1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    cases.append(make_case("transpose", [t1], t1.T))

    # 8. Permute
    p1 = np.arange(1, 7, dtype=np.float32).reshape(1, 2, 3)
    cases.append(make_case("permute", [p1], np.transpose(p1, (2, 0, 1))))

    # Write out as JSON
    with open("test_data/oracles.json", "w") as f:
        json.dump(cases, f, indent=2)

if __name__ == "__main__":
    main()
