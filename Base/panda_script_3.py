# ============================================
# 🎓 Student Performance Predictor
# Built with: Python, Pandas, Scikit-learn
# What it does: Predicts student final grade
# based on study hours, sleep, attendance
# ============================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ── STEP 1: CREATE OUR DATASET ──────────────
# Real world data has many features (columns)
# Each row = one student

data = {
    'study_hours':    [2, 4, 6, 1, 8, 3, 7, 5, 2, 9, 4, 6, 3, 8, 1],
    'sleep_hours':    [6, 7, 8, 5, 8, 6, 7, 7, 4, 9, 6, 8, 5, 7, 6],
    'attendance_pct': [60, 80, 90, 40, 95, 70, 92, 85, 55, 98, 75, 88, 65, 94, 45],
    'final_grade':    [45, 65, 80, 30, 92, 55, 88, 74, 38, 95, 62, 83, 50, 91, 35]
}

df = pd.DataFrame(data)

# ── STEP 2: EXPLORE THE DATA ─────────────────
print("=" * 45)
print("📊 STUDENT DATASET")
print("=" * 45)
print(df)

print("\n📈 BASIC STATISTICS")
print("=" * 45)
print(df.describe().round(2))

# ── STEP 3: SPLIT FEATURES AND LABEL ─────────
# Features = what we know about each student
# Label    = what we want to predict

X = df[['study_hours', 'sleep_hours', 'attendance_pct']]  # 3 features!
y = df['final_grade']

# ── STEP 4: TRAIN / TEST SPLIT ───────────────
# New concept! We split data into two parts:
# 80% for TRAINING  → model learns from this
# 20% for TESTING   → we check if model is accurate
#
# Like studying from 80% of the textbook
# then being tested on questions from the other 20%!

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42     # makes split same every time you run
)

print(f"\n🔀 DATA SPLIT")
print("=" * 45)
print(f"Training samples : {len(X_train)} students")
print(f"Testing samples  : {len(X_test)} students")

# ── STEP 5: TRAIN THE MODEL ───────────────────
model = LinearRegression()
model.fit(X_train, y_train)

# ── STEP 6: TEST THE MODEL ────────────────────
predictions = model.predict(X_test)
accuracy = r2_score(y_test, predictions)

print(f"\n🎯 MODEL ACCURACY")
print("=" * 45)
print(f"Accuracy Score : {accuracy * 100:.2f}%")

# ── STEP 7: PREDICT NEW STUDENTS ─────────────
print(f"\n🔮 PREDICTING NEW STUDENTS")
print("=" * 45)

new_students = pd.DataFrame([
    {'study_hours': 7, 'sleep_hours': 8, 'attendance_pct': 90},
    {'study_hours': 2, 'sleep_hours': 5, 'attendance_pct': 50},
    {'study_hours': 5, 'sleep_hours': 7, 'attendance_pct': 75},
])

for i, student in new_students.iterrows():
    grade = model.predict(pd.DataFrame([student]))[0]
    grade = max(0, min(100, grade))  # keep between 0 and 100

    if grade >= 80:
        emoji = "🌟 Excellent!"
    elif grade >= 60:
        emoji = "✅ Passing"
    elif grade >= 40:
        emoji = "⚠️  At Risk"
    else:
        emoji = "❌ Failing"

    print(f"Student {i+1}: Study={student['study_hours']}h | "
          f"Sleep={student['sleep_hours']}h | "
          f"Attend={student['attendance_pct']}% "
          f"→ Grade: {grade:.1f} {emoji}")

# ── STEP 8: VISUALIZE ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.suptitle('🎓 Student Performance Analysis', fontsize=16, fontweight='bold')

features = ['study_hours', 'sleep_hours', 'attendance_pct']
colors   = ['#3498DB', '#2ECC71', '#E74C3C']
labels   = ['Study Hours', 'Sleep Hours', 'Attendance %']

for i, (feature, color, label) in enumerate(zip(features, colors, labels)):
    axes[i].scatter(df[feature], df['final_grade'],
                    color=color, s=100, alpha=0.7, edgecolors='white')
    axes[i].set_xlabel(label, fontsize=12)
    axes[i].set_ylabel('Final Grade', fontsize=12)
    axes[i].set_title(f'{label} vs Grade', fontsize=13)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Chart saved as results.png!")
print("=" * 45)