"""Quick command-line check:  python predict_cli.py "some news text here" """
import sys

from src.predictor import predict

if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or input("Enter news text: ")
    r = predict(text)
    print(f"{r['label']}  ({r['confidence']}% confident)")
    print(f"Real {r['real_probability']}% | Fake {r['fake_probability']}%")
    if r["top_signals"]:
        print("Key words:", ", ".join(r["top_signals"]))
