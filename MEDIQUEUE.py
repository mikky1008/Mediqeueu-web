import streamlit as st
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timezone, timedelta

# Configure page layout for full width
st.set_page_config(layout="wide")

# Custom CSS for full-width, taller banner
st.markdown("""
<style>
    .st-emotion-cache-1f3p5c4, [data-testid="stAppViewContainer"] h1 {
        width: 100vw;
        margin-left: calc(-50vw + 50%);
        padding: 40px 20px;
        margin-bottom: 20px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Toggle to control whether debug expanders are shown in the UI
# Set to True while developing to see debug panels; keep False in production
SHOW_DEBUG = False

# Define specialties with stable IDs so matching uses numeric ids (keeps compatibility with text values)
SPECIALTIES = [
    (1, "General Practitioner (GP)"),
    (2, "Family Medicine"),
    (3, "Internal Medicine"),
    (4, "Pediatrics"),
    (5, "OB-GYN (Obstetrics & Gynecology)"),
    (6, "Dermatology"),
    (7, "ENT / EENT (Ear, Nose, Throat)"),
    (8, "Ophthalmology (Eye Doctor)"),
    (9, "Cardiology"),
    (10, "Pulmonology"),
    (11, "Gastroenterology"),
    (12, "Endocrinology (Diabetes/Thyroid)"),
    (13, "Psychiatry"),
    (14, "Psychology"),
    (15, "Urology"),
    (16, "Dentistry"),
    (17, "Rehabilitation Medicine / Physical Therapy"),
    (18, "Allergy & Immunology"),
]
SPECIALTY_ID_TO_NAME = {s[0]: s[1] for s in SPECIALTIES}
SPECIALTY_NAME_TO_ID = {s[1]: s[0] for s in SPECIALTIES}

# ----------------------------
# CS 213: Urgency/Priority Levels (for priority queue implementation)
# ----------------------------
URGENCY_LEVELS = [
    (1, "Low - Routine/Preventive"),
    (2, "Medium - Non-urgent but timely"),
    (3, "High - Urgent/Acute symptoms"),
    (4, "Critical - Emergency"),
]
URGENCY_ID_TO_NAME = {u[0]: u[1] for u in URGENCY_LEVELS}
URGENCY_NAME_TO_ID = {u[1]: u[0] for u in URGENCY_LEVELS}

# ----------------------------
# CS 213: Merge Sort Algorithm for Queue Sorting
# ----------------------------
def merge_sort_checkups(checkups, key="urgency", reverse=True):
    """
    Merge sort algorithm to sort checkups by urgency (descending) and then by created_at (ascending).
    CS 213: Efficient O(n log n) sorting for queue organization.
    """
    if len(checkups) <= 1:
        return checkups
    
    mid = len(checkups) // 2
    left = merge_sort_checkups(checkups[:mid], key, reverse)
    right = merge_sort_checkups(checkups[mid:], key, reverse)
    
    return merge_checkups(left, right, key, reverse)

def merge_checkups(left, right, key="urgency", reverse=True):
    """Merge two sorted lists of checkups."""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        left_val = left[i].get(key, 0)
        right_val = right[j].get(key, 0)
        
        # Primary sort by urgency (higher first if reverse=True)
        if reverse:
            if left_val >= right_val:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        else:
            if left_val <= right_val:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
def binary_search_checkup(checkups, checkup_id):
    """Binary search to find a checkup by ID. Requires sorted list."""
    sorted_checkups = sorted(checkups, key=lambda x: x.get("id", 0))
    left, right = 0, len(sorted_checkups) - 1
    
    while left <= right:
        mid = (left + right) // 2
        mid_id = sorted_checkups[mid].get("id", 0)
        
        if mid_id == checkup_id:
            return sorted_checkups[mid]
        elif mid_id < checkup_id:
            left = mid + 1
        else:
            right = mid - 1
    
    return None

# ----------------------------
# CS 213: Calculate Expected Waiting Time (Mathematical Modeling)
# ----------------------------
def calculate_waiting_time(position, avg_consultation_time=15):
    """
    Calculate expected waiting time in minutes.
    Mathematical modeling: position * average consultation time
    """
    return position * avg_consultation_time

# ----------------------------
# CS 213: AVL Tree Implementation for Priority Queue
# ----------------------------
class AVLNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left = None
        self.right = None
        self.height = 1

class AVLTree:
    def __init__(self):
        self.root = None

    def _height(self, node):
        return node.height if node else 0

    def _balance_factor(self, node):
        return self._height(node.left) - self._height(node.right) if node else 0

    def _rotate_right(self, y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        y.height = 1 + max(self._height(y.left), self._height(y.right))
        x.height = 1 + max(self._height(x.left), self._height(x.right))
        return x

    def _rotate_left(self, x):
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        x.height = 1 + max(self._height(x.left), self._height(x.right))
        y.height = 1 + max(self._height(y.left), self._height(y.right))
        return y

    def _insert(self, node, key, value):
        if not node:
            return AVLNode(key, value)
        if key < node.key:
            node.left = self._insert(node.left, key, value)
        else:
            node.right = self._insert(node.right, key, value)

        node.height = 1 + max(self._height(node.left), self._height(node.right))

        balance = self._balance_factor(node)

        # Left Left
        if balance > 1 and key < node.left.key:
            return self._rotate_right(node)
        # Right Right
        if balance < -1 and key > node.right.key:
            return self._rotate_left(node)
        # Left Right
        if balance > 1 and key > node.left.key:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        # Right Left
        if balance < -1 and key < node.right.key:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node

    def insert(self, key, value):
        self.root = self._insert(self.root, key, value)

    def inorder(self):
        result = []

        def _inorder(node):
            if not node:
                return
            _inorder(node.left)
            result.append(node.value)
            _inorder(node.right)

        _inorder(self.root)
        return result


def build_avl_from_checkups(checkups):
    """
    Build an AVL tree from checkups using a composite key:
    (-urgency, created_at_timestamp, id) so higher urgency comes first,
    then earlier created_at.
    Returns the AVLTree instance.
    """
    tree = AVLTree()
    for c in checkups:
        urgency = int(c.get("urgency", 2) or 2)
        created = c.get("created_at")
        # Normalize created to timestamp for comparison
        try:
            if isinstance(created, datetime):
                ts = created.timestamp()
            else:
                ts = float(created) if created is not None else 0.0
        except Exception:
            ts = 0.0

        cid = c.get("id") or 0
        sort_key = (-urgency, ts, cid)
        tree.insert(sort_key, c)

    return tree

# ----------------------------
# CS 213: Quicksort Implementation (alternative to merge sort)
# ----------------------------
def quick_sort_checkups(checkups, key="urgency", reverse=True):
    if len(checkups) <= 1:
        return checkups
    pivot = checkups[len(checkups) // 2]
    pivot_val = pivot.get(key, 0)
    left = [x for x in checkups if (x.get(key, 0) > pivot_val) == reverse]
    middle = [x for x in checkups if x.get(key, 0) == pivot_val]
    right = [x for x in checkups if (x.get(key, 0) < pivot_val) == reverse]
    return quick_sort_checkups(left, key, reverse) + middle + quick_sort_checkups(right, key, reverse)


# ----------------------------
# CS 213: Unified sort wrapper (default: AVL)
# ----------------------------
def sort_checkups(checkups, method="avl", key="urgency", reverse=True):
    """
    Wrapper to sort checkups using chosen method.
    method: "avl", "merge", "quick"
    """
    if not checkups:
        return checkups
    try:
        if method == "avl":
            tree = build_avl_from_checkups(checkups)
            return tree.inorder()
        elif method == "quick":
            return quick_sort_checkups(checkups, key=key, reverse=reverse)
        else:
            return merge_sort_checkups(checkups, key=key, reverse=reverse)
    except Exception:
        return merge_sort_checkups(checkups, key=key, reverse=reverse)

# ----------------------------
# CS 213: Linked List + Department Queue Manager
# ----------------------------
class ListNode:
    def __init__(self, checkup):
        self.checkup = checkup
        self.next = None


class LinkedList:
    """Simple singly linked list to represent a queue."""
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def insert_end(self, checkup):
        node = ListNode(checkup)
        if not self.head:
            self.head = self.tail = node
        else:
            self.tail.next = node
            self.tail = node
        self.size += 1

    def remove_by_id(self, checkup_id):
        prev = None
        cur = self.head
        while cur:
            cid = cur.checkup.get("id") or cur.checkup.get("patient_id")
            if cid == checkup_id:
                if prev:
                    prev.next = cur.next
                else:
                    self.head = cur.next
                if cur is self.tail:
                    self.tail = prev
                self.size -= 1
                return cur.checkup
            prev = cur
            cur = cur.next
        return None

    def to_list(self):
        out = []
        cur = self.head
        while cur:
            out.append(cur.checkup)
            cur = cur.next
        return out

    def find(self, checkup_id):
        cur = self.head
        while cur:
            cid = cur.checkup.get("id") or cur.checkup.get("patient_id")
            if cid == checkup_id:
                return cur.checkup
            cur = cur.next
        return None


class DepartmentQueues:
    """
    Manage queues for multiple hospital departments.
    Supports array-backed queues, linked-list queues, and heap priority queues.
    """
    def __init__(self):
        # dict: dept_name -> {'array': [...], 'linked': LinkedList(), 'heap': heap}
        self.queues = {}

    def ensure_dept(self, dept_name):
        if dept_name not in self.queues:
            self.queues[dept_name] = {
                'array': [],
                'linked': LinkedList(),
                'heap': []
            }

    def enqueue(self, dept_name, checkup):
        self.ensure_dept(dept_name)
        # array
        self.queues[dept_name]['array'].append(checkup)
        # linked list
        self.queues[dept_name]['linked'].insert_end(checkup)
        # heap (priority)
        try:
            urgency = int(checkup.get('urgency', 2) or 2)
        except Exception:
            urgency = 2
        # use negative urgency for max-heap behavior with heapq
        heapq.heappush(self.queues[dept_name]['heap'], (-urgency, checkup.get('id', 0), checkup))

    def dequeue_array(self, dept_name):
        self.ensure_dept(dept_name)
        if self.queues[dept_name]['array']:
            item = self.queues[dept_name]['array'].pop(0)
            # remove from linked and heap lazily
            self.queues[dept_name]['linked'].remove_by_id(item.get('id'))
            # heap lazy removal: not implemented (pop when matching top)
            return item
        return None

    def dequeue_heap(self, dept_name):
        self.ensure_dept(dept_name)
        if not self.queues[dept_name]['heap']:
            return None
        _, _, checkup = heapq.heappop(self.queues[dept_name]['heap'])
        # remove from array and linked list
        try:
            self.queues[dept_name]['array'].remove(checkup)
        except Exception:
            pass
        try:
            self.queues[dept_name]['linked'].remove_by_id(checkup.get('id'))
        except Exception:
            pass
        return checkup

    def move_patient(self, checkup_id, from_dept, to_dept):
        self.ensure_dept(from_dept)
        self.ensure_dept(to_dept)
        # try linked removal first
        checkup = self.queues[from_dept]['linked'].remove_by_id(checkup_id)
        if not checkup:
            # try array
            arr = self.queues[from_dept]['array']
            for i, c in enumerate(arr):
                cid = c.get('id') or c.get('patient_id')
                if cid == checkup_id:
                    checkup = arr.pop(i)
                    break
        if not checkup:
            # heap: scan and remove
            heap = self.queues[from_dept]['heap']
            for i, item in enumerate(heap):
                if item[2].get('id') == checkup_id:
                    checkup = item[2]
                    del heap[i]
                    heapq.heapify(heap)
                    break
        if checkup:
            self.enqueue(to_dept, checkup)
            return True
        return False

    def get_queue_snapshot(self, dept_name, method='array'):
        self.ensure_dept(dept_name)
        if method == 'linked':
            return self.queues[dept_name]['linked'].to_list()
        elif method == 'heap':
            return [item[2] for item in sorted(self.queues[dept_name]['heap'], reverse=False)]
        else:
            return list(self.queues[dept_name]['array'])

    def build_from_checkups(self, checkups, dept_key='specialty'):
        """Partition checkups into department queues by `dept_key` field."""
        for c in checkups:
            dept = c.get(dept_key) or c.get('preferred_specialty') or 'General'
            # normalize dept name if numeric
            try:
                if isinstance(dept, int):
                    dept = SPECIALTY_ID_TO_NAME.get(dept, str(dept))
                else:
                    # map known names to themselves
                    dept = str(dept)
            except Exception:
                dept = str(dept)
            self.enqueue(dept, c)

# ----------------------------
# CS 213: Linear Search for Patient ID Lookup
# ----------------------------
def linear_search_patient_id(patients, patient_id):
    """
    Linear search to check if a patient ID already exists.
    CS 213: Simple O(n) search for patient registration validation.
    """
    for patient in patients:
        if patient.get("id") == patient_id or patient.get("patient_id") == patient_id:
            return patient
    return None


# ----------------------------
# Collision management for queue
# ----------------------------
def detect_checkup_collision(existing_checkups, new_checkup):
    """
    Detect potential collisions between a new checkup and an existing list of checkups.

    Collisions detected:
      - 'patient': same patient_id (duplicate checkup)
      - 'slot': same doctor + appointment_date + appointment_time
      - 'contact': same contact at same datetime

    Returns a list of tuples: (collision_type, existing_checkup)
    """
    conflicts = []
    try:
        new_pid = new_checkup.get("patient_id")
        new_doc = new_checkup.get("doctor_id")
        new_date = new_checkup.get("appointment_date")
        new_time = new_checkup.get("appointment_time")
        new_contact = (new_checkup.get("contact") or "").strip()

        for c in existing_checkups:
            # patient duplicate
            if new_pid and c.get("patient_id") and str(c.get("patient_id")) == str(new_pid):
                conflicts.append(("patient", c))
                continue

            # exact slot collision (same doctor, date, time)
            if new_doc and c.get("doctor_id") and new_date and new_time:
                if (str(c.get("doctor_id")) == str(new_doc)
                        and str(c.get("appointment_date")) == str(new_date)
                        and str(c.get("appointment_time")) == str(new_time)):
                    conflicts.append(("slot", c))
                    continue

            # contact-based duplicate at same slot
            if new_contact and c.get("contact") and new_date and new_time:
                if (c.get("contact").strip() == new_contact
                        and str(c.get("appointment_date")) == str(new_date)
                        and str(c.get("appointment_time")) == str(new_time)):
                    conflicts.append(("contact", c))
                    continue
    except Exception:
        # Best-effort: if any error happens, return empty conflicts rather than crash
        return []

    return conflicts


def resolve_checkup_collision(existing_checkups, new_checkup, strategy="auto"):
    """
    Resolve collisions according to strategy.

    strategy:
      - 'auto': merge if same patient_id, otherwise reject slot collisions
      - 'merge': merge into existing (combine symptoms, keep higher urgency)
      - 'reject': reject new checkup

    Returns a dict: {'accepted': bool, 'action': str, 'existing': existing_checkup_or_None, 'message': str}
    """
    conflicts = detect_checkup_collision(existing_checkups, new_checkup)
    if not conflicts:
        return {"accepted": True, "action": "insert", "existing": None, "message": "No conflict"}

    # Prefer patient collisions for merging
    for kind, existing in conflicts:
        if kind == "patient":
            if strategy in ("auto", "merge"):
                # Merge symptoms and urgency
                try:
                    # Merge symptoms text
                    new_sym = (new_checkup.get("symptoms") or "").strip()
                    old_sym = (existing.get("symptoms") or "").strip()
                    if new_sym and new_sym not in old_sym:
                        existing["symptoms"] = (old_sym + "\n" + new_sym).strip()

                    # Keep the highest urgency (if numeric)
                    try:
                        existing_urg = int(existing.get("urgency", 2) or 2)
                    except Exception:
                        existing_urg = 2
                    try:
                        new_urg = int(new_checkup.get("urgency", 2) or 2)
                    except Exception:
                        new_urg = 2
                    existing["urgency"] = max(existing_urg, new_urg)

                    return {"accepted": True, "action": "merged", "existing": existing, "message": "Merged with existing patient checkup"}
                except Exception as e:
                    return {"accepted": False, "action": "error", "existing": existing, "message": f"Merge failed: {e}"}

    # If no patient collision, handle slot/contact collisions
    for kind, existing in conflicts:
        if kind in ("slot", "contact"):
            if strategy == "reject" or strategy == "auto":
                return {"accepted": False, "action": "reject", "existing": existing, "message": f"Slot conflict ({kind}) with existing booking"}
            if strategy == "merge":
                # merging slot collisions may be undesirable; fallback to reject
                return {"accepted": False, "action": "reject", "existing": existing, "message": "Cannot merge slot collision"}

    return {"accepted": False, "action": "unknown", "existing": conflicts[0][1], "message": "Unhandled conflict"}


def safe_enqueue_checkup(checkups_list, new_checkup, strategy="auto"):
    """
    Safely enqueue a checkup into an in-memory list with collision handling.

    Returns: (success: bool, result: dict)
    """
    res = resolve_checkup_collision(checkups_list, new_checkup, strategy=strategy)
    if not res.get("accepted"):
        return False, res

    action = res.get("action")
    if action == "insert":
        checkups_list.append(new_checkup)
        return True, {"action": "inserted", "message": "Checkup added"}
    if action == "merged":
        # existing checkup already updated in-place
        return True, {"action": "merged", "message": "Checkup merged with existing"}

    return False, {"action": "unknown", "message": "Could not enqueue checkup"}


# ----------------------------
# CS 213: Hash-Based Patient Dictionary for O(1) Lookup
# ----------------------------
def build_patient_hash_map(patients):
    """
    Build a hash map (dictionary) for O(1) patient lookup by ID.
    CS 213: Hashing for quick access to patient information.
    """
    patient_map = {}
    for patient in patients:
        patient_id = patient.get("id") or patient.get("patient_id")
        if patient_id:
            patient_map[patient_id] = patient
    return patient_map

def get_patient_by_id_hash(patient_map, patient_id):
    """Retrieve patient from hash map in O(1) time."""
    return patient_map.get(patient_id)

# ----------------------------
# CS 213: Heap/Priority Queue Operations
# ----------------------------
import heapq

def build_priority_queue(checkups):
    """
    Build a min-heap priority queue from checkups.
    CS 213: Heap operations for efficient priority queue management.
    Lower urgency numbers = higher priority (Critical=4 treated first)
    """
    # Negate urgency to create max-heap behavior (higher urgency first)
    heap = []
    for idx, checkup in enumerate(checkups):
        urgency = checkup.get("urgency", 2)
        # Tuple: (negative_urgency, index, checkup_data)
        # Negative urgency: 4 (Critical) becomes -4, so it's popped first
        heapq.heappush(heap, (-urgency, idx, checkup))
    return heap

def pop_next_patient(heap):
    """Pop the highest priority (most urgent) patient from the heap."""
    if not heap:
        return None
    neg_urgency, idx, checkup = heapq.heappop(heap)
    return checkup

def peek_next_patient(heap):
    """Peek at the highest priority patient without removing from heap."""
    if not heap:
        return None
    neg_urgency, idx, checkup = heap[0]
    return checkup

# ----------------------------
# CS 213: Greedy Algorithm for Next Patient Selection
# ----------------------------
def select_next_patient_greedy(checkups):
    """
    Greedy algorithm: Always select the patient with highest urgency.
    CS 213: Greedy approach for optimal next-patient assignment.
    Returns the most urgent patient not yet processed.
    """
    if not checkups:
        return None
    
    # Find patient with maximum urgency
    max_urgency_patient = max(checkups, key=lambda x: x.get("urgency", 2))
    return max_urgency_patient

def select_next_patients_by_urgency(checkups, num_patients=5):
    """
    Greedy selection of top N patients by urgency.
    CS 213: Multi-patient greedy selection for batch processing.
    """
    if not checkups:
        return []
    
    sorted_by_urgency = sorted(checkups, key=lambda x: x.get("urgency", 2), reverse=True)
    return sorted_by_urgency[:num_patients]

# ----------------------------
# CS 213: Poisson Distribution for Patient Arrivals
# ----------------------------
import math

def poisson_probability(lambda_param, k):
    """
    Calculate Poisson probability: P(X = k) for arrival rate lambda_param.
    CS 213: Discrete mathematics for predicting patient arrivals.
    
    lambda_param: Expected number of arrivals per time period
    k: Number of arrivals we want probability for
    
    Formula: P(X=k) = (e^-λ * λ^k) / k!
    """
    if lambda_param <= 0:
        return 0.0

    # Compute in log-space to avoid overflow for large k
    try:
        # log P = -lambda + k*log(lambda) - log(k!)
        log_p = -lambda_param + k * math.log(lambda_param) - math.lgamma(k + 1)
        return math.exp(log_p)
    except (OverflowError, ValueError):
        # Underflow/overflow -> probability effectively 0
        return 0.0

def poisson_cumulative(lambda_param, k):
    """
    Calculate cumulative Poisson probability: P(X <= k).
    Useful for determining probability that arrivals won't exceed k.
    """
    # For non-positive lambda, only P(X=0)=1 when lambda==0
    if lambda_param <= 0:
        return 1.0 if k >= 0 else 0.0

    total = 0.0
    for i in range(int(k) + 1):
        total += poisson_probability(lambda_param, i)
    # Clamp to [0,1]
    return min(1.0, max(0.0, total))

def expected_arrivals_per_period(historical_arrival_rate, time_periods):
    """
    Calculate expected patient arrivals for given time periods.
    lambda_param = (average arrivals per hour) * (number of hours)
    """
    return historical_arrival_rate * time_periods

# ----------------------------
# CS 213: Statistical Analysis for Queue Management
# ----------------------------
def calculate_average_wait_time(checkups):
    """Calculate average waiting time across all patients in queue."""
    if not checkups:
        return 0
    
    total_wait = 0
    for idx, checkup in enumerate(checkups):
        total_wait += calculate_waiting_time(idx + 1)
    
    return total_wait / len(checkups)

def calculate_queue_statistics(checkups, avg_consultation_time=15):
    """
    Calculate comprehensive queue statistics.
    CS 213: Statistical analysis for performance evaluation.
    
    Returns dict with:
    - total_patients: Number of patients in queue
    - average_wait_time: Expected average wait time
    - max_wait_time: Wait time for last patient
    - urgency_distribution: Count of each urgency level
    - total_urgent_patients: Count of High+Critical patients
    """
    if not checkups:
        return {
            "total_patients": 0,
            "average_wait_time": 0,
            "max_wait_time": 0,
            "urgency_distribution": {1: 0, 2: 0, 3: 0, 4: 0},
            "total_urgent_patients": 0,
            "queue_efficiency": 100.0
        }
    
    # Count urgency levels
    urgency_dist = {1: 0, 2: 0, 3: 0, 4: 0}
    for checkup in checkups:
        urgency = checkup.get("urgency", 2)
        urgency_dist[urgency] = urgency_dist.get(urgency, 0) + 1
    
    total_patients = len(checkups)
    max_wait = calculate_waiting_time(total_patients, avg_consultation_time)
    avg_wait = calculate_average_wait_time(checkups)
    
    # Patients with high or critical urgency
    urgent_count = urgency_dist.get(3, 0) + urgency_dist.get(4, 0)
    
    # Queue efficiency: percentage of patients in queue (vs completed)
    # This can be calculated based on historical data
    queue_efficiency = min(100.0, (total_patients / 50.0) * 100) if total_patients > 0 else 0
    
    return {
        "total_patients": total_patients,
        "average_wait_time": round(avg_wait, 2),
        "max_wait_time": round(max_wait, 2),
        "urgency_distribution": urgency_dist,
        "total_urgent_patients": urgent_count,
        "queue_efficiency": round(queue_efficiency, 2)
    }

def predict_queue_overflow(current_queue_size, arrival_rate, max_capacity=100):
    """
    Predict if queue will overflow based on arrival rate.
    Uses Poisson distribution to forecast.
    
    Returns probability that queue will exceed max_capacity within 1 hour.
    """
    # Expected arrivals in next hour
    expected_arrivals = expected_arrivals_per_period(arrival_rate, 1)
    projected_queue = current_queue_size + expected_arrivals
    
    # Probability of overflow
    overflow_prob = 1 - poisson_cumulative(arrival_rate, max_capacity - current_queue_size)
    
    return {
        "current_queue": current_queue_size,
        "expected_arrivals": round(expected_arrivals, 2),
        "projected_queue_size": round(projected_queue, 2),
        "max_capacity": max_capacity,
        "overflow_probability": round(overflow_prob * 100, 2),
        "will_overflow": projected_queue > max_capacity
    }

def calculate_service_level(checkups, max_acceptable_wait=60):
    """
    Calculate service level: percentage of patients with acceptable wait time.
    CS 213: Performance metrics using discrete mathematics.
    
    max_acceptable_wait: Maximum acceptable wait time in minutes
    """
    if not checkups:
        return 100.0
    
    acceptable_count = 0
    for idx, checkup in enumerate(checkups):
        wait_time = calculate_waiting_time(idx + 1)
        if wait_time <= max_acceptable_wait:
            acceptable_count += 1
    
    return round((acceptable_count / len(checkups)) * 100, 2)

# ----------------------------
# CS 213: Advanced Discrete Mathematics - Patient Flow Prediction
# ----------------------------
def forecast_cancellation_probability(arrival_rate, historical_cancellation_rate=0.05):
    """
    Forecast cancellation probability using discrete probability.
    CS 213: Discrete mathematics for predicting patient cancellations.
    
    arrival_rate: Expected patients per hour
    historical_cancellation_rate: Historical cancellation percentage (0.0-1.0)
    
    Returns: Probability that at least one patient cancels in next hour.
    """
    # Using Poisson for arrivals, then multiply by cancellation rate
    expected_arrivals = arrival_rate
    expected_cancellations = expected_arrivals * historical_cancellation_rate
    
    # Probability of at least one cancellation
    # P(X >= 1) = 1 - P(X = 0)
    prob_no_cancellation = math.exp(-expected_cancellations)
    prob_at_least_one_cancellation = 1 - prob_no_cancellation
    
    return round(prob_at_least_one_cancellation * 100, 2)


def forecast_late_arrivals(arrival_rate, historical_late_rate=0.10):
    """
    Forecast late arrival probability.
    CS 213: Discrete probability for predicting late arrivals.
    
    arrival_rate: Expected patients per hour
    historical_late_rate: Historical late arrival percentage (0.0-1.0)
    
    Returns: Probability that patients will arrive late in next hour.
    """
    expected_arrivals = arrival_rate
    expected_late = expected_arrivals * historical_late_rate
    
    # P(at least one late arrival) = 1 - P(no late arrivals)
    prob_no_late = math.exp(-expected_late)
    prob_late = 1 - prob_no_late
    
    return round(prob_late * 100, 2)


def calculate_crowding_index(queue_size, max_comfortable_capacity=30, max_safe_capacity=100):
    """
    Calculate physical crowding index (0-100).
    CS 213: Discrete metrics for assessing facility crowding.
    
    queue_size: Current number of patients waiting
    max_comfortable_capacity: Comfortable operating capacity
    max_safe_capacity: Maximum safe capacity
    
    Returns: Crowding index (0=empty, 100=overcrowded/unsafe)
    """
    if queue_size <= 0:
        return 0.0
    elif queue_size >= max_safe_capacity:
        return 100.0
    elif queue_size >= max_comfortable_capacity:
        # Between comfortable and safe: scale 50-100
        ratio = (queue_size - max_comfortable_capacity) / (max_safe_capacity - max_comfortable_capacity)
        return round(50 + (ratio * 50), 2)
    else:
        # Below comfortable: scale 0-50
        ratio = queue_size / max_comfortable_capacity
        return round(ratio * 50, 2)


def estimate_patient_satisfaction(avg_wait_time, urgency_level=2):
    """
    Estimate patient satisfaction based on wait time and urgency.
    CS 213: Statistical correlation between wait time and satisfaction.
    
    avg_wait_time: Average wait time in minutes
    urgency_level: Patient urgency (1=Low, 2=Medium, 3=High, 4=Critical)
    
    Returns: Satisfaction score (0-100, where 100 is very satisfied)
    """
    # Base satisfaction by urgency (critical patients expect shorter waits)
    urgency_expectations = {
        1: 90,   # Low: very tolerant
        2: 60,   # Medium: moderate tolerance
        3: 30,   # High: low tolerance
        4: 10    # Critical: expects immediate care
    }
    
    base_satisfaction = urgency_expectations.get(urgency_level, 50)
    
    # Reduce satisfaction based on wait time
    # Each additional 10 minutes reduces satisfaction by ~5%
    wait_penalty = min(base_satisfaction, (avg_wait_time / 10) * 5)
    
    satisfaction = max(0, base_satisfaction - wait_penalty)
    return round(satisfaction, 2)


def calculate_improvement_metrics(previous_stats, current_stats):
    """
    Calculate improvement metrics comparing previous period to current.
    CS 213: Statistical analysis for evaluating system improvements.
    
    previous_stats, current_stats: dicts with keys:
    - 'avg_wait_time', 'total_patients', 'service_level', 'satisfaction_avg'
    
    Returns: dict with percentage improvements and trends.
    """
    improvements = {}
    
    # Wait time improvement (negative = improvement)
    if previous_stats.get('avg_wait_time', 0) > 0:
        wait_improvement = ((previous_stats['avg_wait_time'] - current_stats.get('avg_wait_time', 0)) 
                            / previous_stats['avg_wait_time']) * 100
        improvements['wait_time_improvement_percent'] = round(wait_improvement, 2)
    
    # Service level improvement
    prev_service = previous_stats.get('service_level', 0)
    curr_service = current_stats.get('service_level', 0)
    if prev_service > 0:
        service_improvement = ((curr_service - prev_service) / prev_service) * 100
        improvements['service_level_improvement_percent'] = round(service_improvement, 2)
    
    # Patient satisfaction improvement
    prev_satisfaction = previous_stats.get('satisfaction_avg', 50)
    curr_satisfaction = current_stats.get('satisfaction_avg', 50)
    satisfaction_improvement = ((curr_satisfaction - prev_satisfaction) / max(prev_satisfaction, 1)) * 100
    improvements['satisfaction_improvement_percent'] = round(satisfaction_improvement, 2)
    
    # Throughput (patients served per time period)
    improvements['current_throughput'] = current_stats.get('total_patients', 0)
    improvements['previous_throughput'] = previous_stats.get('total_patients', 0)
    
    # Overall trend
    if improvements['wait_time_improvement_percent'] > 0:
        improvements['trend'] = "↓ Wait times improving"
    elif improvements['wait_time_improvement_percent'] < 0:
        improvements['trend'] = "↑ Wait times increasing"
    else:
        improvements['trend'] = "→ Wait times stable"
    
    return improvements


def forecast_patient_load(historical_data, forecast_hours=1):
    """
    Forecast expected patient load for next N hours.
    CS 213: Poisson-based forecasting for patient arrivals.
    
    historical_data: list of dicts with 'timestamp' and 'arrivals' count
    forecast_hours: Number of hours to forecast
    
    Returns: dict with mean forecast, confidence intervals, and probabilities.
    """
    if not historical_data:
        return {'mean': 0, 'lower_bound': 0, 'upper_bound': 0}
    
    # Calculate average arrival rate
    total_arrivals = sum(d.get('arrivals', 0) for d in historical_data)
    periods = max(len(historical_data), 1)
    arrival_rate = total_arrivals / periods
    
    # Scale by forecast hours
    expected_arrivals = arrival_rate * forecast_hours
    
    # 95% confidence interval for Poisson (approximately normal for large lambda)
    std_dev = math.sqrt(expected_arrivals)
    lower_bound = max(0, expected_arrivals - (1.96 * std_dev))
    upper_bound = expected_arrivals + (1.96 * std_dev)
    
    return {
        'mean': round(expected_arrivals, 2),
        'lower_bound': round(lower_bound, 2),
        'upper_bound': round(upper_bound, 2),
        'std_dev': round(std_dev, 2)
    }

# ----------------------------
# Appointment & Cancellation Functions
# ----------------------------
def can_cancel_appointment(appointment_date, appointment_time):
    """
    Check if appointment can be cancelled (must be 12+ hours away).
    
    Returns: (can_cancel: bool, hours_remaining: float, message: str)
    """
    # Validate inputs first
    if not appointment_date or not appointment_time:
        return False, 0, "Appointment date/time not set"

    try:
        # Normalize appointment_date
        if isinstance(appointment_date, datetime):
            appointment_date = appointment_date.date()
        elif isinstance(appointment_date, str):
            try:
                appointment_date = datetime.strptime(appointment_date, '%Y-%m-%d').date()
            except Exception:
                # Try ISO format fallback
                appointment_date = datetime.fromisoformat(appointment_date).date()

        # Normalize appointment_time
        if isinstance(appointment_time, datetime):
            appointment_time = appointment_time.time()
        elif isinstance(appointment_time, str):
            # Accept 'HH:MM' or 'HH:MM:SS'
            parsed_time = None
            for fmt in ('%H:%M:%S', '%H:%M'):
                try:
                    parsed_time = datetime.strptime(appointment_time, fmt).time()
                    break
                except Exception:
                    continue
            if parsed_time is None:
                # Try ISO time
                appointment_time = datetime.fromisoformat(appointment_time).time()
            else:
                appointment_time = parsed_time

        # Combine and compute hours remaining
        appointment_datetime = datetime.combine(appointment_date, appointment_time)
        now = datetime.now()

        time_diff = appointment_datetime - now
        hours_remaining = time_diff.total_seconds() / 3600

        can_cancel = hours_remaining >= 12

        if can_cancel:
            message = f"{hours_remaining:.1f} hours remaining - cancellation available"
        else:
            message = f"Only {max(hours_remaining, 0):.1f} hours remaining - cancellation window closed"

        return can_cancel, hours_remaining, message
    except Exception as e:
        # Return a friendly message rather than raising
        return False, 0, f"Error: {e}"


def can_cancel_from_checkin(created_at):
    """
    Allow cancellation for up to 12 hours after the patient checked in (created_at).

    Returns: (can_cancel: bool, hours_remaining: float, message: str)
    """
    if not created_at:
        return True, 12.0, "Check-in time not set, cancellation allowed"

    try:
        ts = created_at
        # Parse common string formats if needed
        if isinstance(created_at, str):
            try:
                ts = datetime.fromisoformat(created_at)
            except Exception:
                try:
                    ts = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    try:
                        ts = datetime.strptime(created_at, "%Y-%m-%d %H:%M")
                    except Exception:
                        # last resort: treat as naive now-0 to allow cancellation
                        return True, 12.0, "Unrecognized check-in time format; cancellation allowed"

        # Determine current time matching tz-awareness of ts
        if getattr(ts, 'tzinfo', None):
            now = datetime.now(timezone.utc)
        else:
            now = datetime.now()

        elapsed_hours = (now - ts).total_seconds() / 3600.0
        hours_remaining = max(0.0, 12.0 - elapsed_hours)
        can_cancel = elapsed_hours < 12.0

        if can_cancel:
            msg = f"{hours_remaining:.1f} hours left to cancel (from check-in)"
        else:
            msg = f"Cancellation window (12h) expired"

        return can_cancel, hours_remaining, msg
    except Exception as e:
        return False, 0.0, f"Error determining cancel window: {e}"

def cancel_checkup(checkup_id, patient_id):
    """
    Cancel a checkup appointment.
    
    Returns: (success: bool, message: str)
    """
    conn = get_connection()
    if not conn:
        return False, "Database connection failed"
    
    try:
        cursor = conn.cursor()
        
        # Verify checkup belongs to this patient
        cursor.execute(
            "SELECT id FROM checkups WHERE id=%s AND patient_id=%s",
            (checkup_id, patient_id)
        )
        
        if not cursor.fetchone():
            return False, "Checkup not found or doesn't belong to you"
        
        # Delete the checkup
        cursor.execute(
            "DELETE FROM checkups WHERE id=%s AND patient_id=%s",
            (checkup_id, patient_id)
        )
        conn.commit()
        
        return True, "Checkup cancelled successfully"
    except Error as e:
        return False, f"Database error: {e}"
    finally:
        cursor.close()
        conn.close()

def get_patient_checkups(patient_id):
    """
    Retrieve all upcoming checkups for a patient.
    
    Returns: list of checkup dicts
    """
    conn = get_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                c.id,
                c.patient_id,
                c.fullname,
                c.age,
                c.contact,
                c.symptoms,
                c.appointment_date,
                c.appointment_time,
                c.doctor_id,
                c.urgency,
                c.created_at,
                d.firstname,
                d.lastname,
                d.specialty
            FROM checkups c
            LEFT JOIN doctors d ON c.doctor_id = d.id
            WHERE c.patient_id = %s
            ORDER BY c.appointment_date ASC, c.appointment_time ASC
        """, (patient_id,))
        
        checkups = cursor.fetchall()
        return checkups if checkups else []
    except Error as e:
        st.error(f"Error retrieving checkups: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

# ----------------------------
# Database connection
# ----------------------------
def get_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="medqueue"
        )
        return conn
    except Error as e:
        st.error(f"Error connecting to database: {e}")
        return None

def ensure_appointment_date_column():
    """Ensure appointment_date column exists in checkups table."""
    conn = get_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Check if appointment_date column exists
        cursor.execute("""
            SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME='checkups' AND COLUMN_NAME='appointment_date'
        """)
        
        if not cursor.fetchone():
            # Column doesn't exist, add it
            cursor.execute("""
                ALTER TABLE checkups 
                ADD COLUMN appointment_date DATE DEFAULT NULL
            """)
            conn.commit()
        
        cursor.close()
        conn.close()
        return True
    except Error as e:
        st.warning(f"Could not ensure appointment_date column: {e}")
        return False

def ensure_history_table():
    """Create checkups_history table if it doesn't exist, based on checkups table structure."""
    conn = get_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # First, try to create checkups_history by copying the checkups structure
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkups_history LIKE checkups
        """)
        
        # Add finished_at column if it doesn't exist
        try:
            cursor.execute("""
                ALTER TABLE checkups_history 
                ADD COLUMN finished_at DATETIME DEFAULT CURRENT_TIMESTAMP
            """)
        except Exception:
            # Column already exists, that's fine
            pass
        
        # Add doctor_id column if it doesn't exist
        try:
            cursor.execute("""
                ALTER TABLE checkups_history 
                ADD COLUMN doctor_id INT
            """)
        except Exception:
            # Column already exists, that's fine
            pass
        
        # Remove AUTO_INCREMENT from the id column in history table to avoid conflicts
        # This allows us to explicitly insert IDs from the original checkups
        try:
            cursor.execute("""
                ALTER TABLE checkups_history MODIFY COLUMN id INT NOT NULL
            """)
        except Exception:
            # Column might not exist or already be modified, that's fine
            pass
        
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error creating history table: {e}")
        return False
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

def ensure_appointment_time_column():
    """Add appointment_time column to checkups table if it doesn't exist."""
    conn = get_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                ALTER TABLE checkups 
                ADD COLUMN appointment_time TIME DEFAULT NULL
            """)
            conn.commit()
        except Exception:
            # Column already exists, that's fine
            pass
        return True
    except Exception as e:
        return False
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

def ensure_urgency_column():
    """Add urgency column to checkups table if it doesn't exist. (CS 213: Priority Queue)"""
    conn = get_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                ALTER TABLE checkups 
                ADD COLUMN urgency INT DEFAULT 2
            """)
            conn.commit()
        except Exception:
            # Column already exists, that's fine
            pass
        return True
    except Exception as e:
        return False
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

# Password hashing (kept simple here; replace with proper hashing in production)
def hash_password(password):
    return password

# Safe rerun helper: some Streamlit versions may not expose
# `st.experimental_rerun` directly; try it, fall back to raising
# the internal RerunException, and finally `st.stop()` as a last resort.
def safe_rerun():
    try:
        # Preferred public API (most versions)
        st.experimental_rerun()
        return
    except Exception:
        pass

    try:
        # Internal RerunException used by Streamlit's runtime
        from streamlit.runtime.scriptrunner.script_runner import RerunException
        raise RerunException()
    except Exception:
        # Best-effort fallback to stop execution (no immediate rerun)
        st.stop()

# ----------------------------
# Auth: signup & login
# ----------------------------
def signup(user_type, firstname, middlename, lastname, email, contact, password, specialty=None):
    conn = get_connection()
    if conn:
        try:
            cursor = conn.cursor(buffered=True)
            hashed_pw = hash_password(password)
            email_norm = email.strip().lower()
            contact_norm = contact.strip()

            # Ensure uniqueness of email and contact across both doctors and patients
            try:
                # Check email
                cursor.execute("SELECT EXISTS(SELECT 1 FROM doctors WHERE email=%s)", (email_norm,))
                exists_doc_email = bool(cursor.fetchone()[0])
                cursor.execute("SELECT EXISTS(SELECT 1 FROM patients WHERE email=%s)", (email_norm,))
                exists_pat_email = bool(cursor.fetchone()[0])
                if exists_doc_email or exists_pat_email:
                    st.error("An account with this email already exists. Please use a different email or log in.")
                    return False

                # Check contact number
                if contact_norm:
                    cursor.execute("SELECT EXISTS(SELECT 1 FROM doctors WHERE contact=%s)", (contact_norm,))
                    exists_doc_contact = bool(cursor.fetchone()[0])
                    cursor.execute("SELECT EXISTS(SELECT 1 FROM patients WHERE contact=%s)", (contact_norm,))
                    exists_pat_contact = bool(cursor.fetchone()[0])
                    if exists_doc_contact or exists_pat_contact:
                        st.error("An account with this contact number already exists. Please use a different contact number.")
                        return False
            except Error:
                # If uniqueness checks fail (missing tables/columns), continue and rely on DB constraints if present
                pass

            if user_type == 'Doctor':
                # include specialty column for doctors (specialty may be None)
                try:
                    cursor.execute(
                        f"INSERT INTO doctors (firstname, middlename, lastname, email, contact, password, specialty) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        (firstname.strip(), middlename.strip(), lastname.strip(), email_norm, contact.strip(), hashed_pw, specialty)
                    )
                except Error:
                    # Fallback if the doctors table doesn't have a 'specialty' column
                    cursor.execute(
                        f"INSERT INTO doctors (firstname, middlename, lastname, email, contact, password) VALUES (%s, %s, %s, %s, %s, %s)",
                        (firstname.strip(), middlename.strip(), lastname.strip(), email_norm, contact.strip(), hashed_pw)
                    )
            else:
                cursor.execute(
                    f"INSERT INTO patients (firstname, middlename, lastname, email, contact, password) VALUES (%s, %s, %s, %s, %s, %s)",
                    (firstname.strip(), middlename.strip(), lastname.strip(), email_norm, contact.strip(), hashed_pw)
                )

            conn.commit()
            st.success(f"{user_type} account created successfully!")
            return True

        except Error as e:
            st.error(f"Error during signup: {e}")
            return False

        finally:
            try:
                cursor.close()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
    return False

def login(user_type, email, password):
    """
    Returns a dictionary-like user record (if found) or None.
    Keeps original behavior but tries to return a dict when possible.
    """
    conn = get_connection()
    if conn:
        try:
            # dictionary cursor so keys are available
            cursor = conn.cursor(dictionary=True, buffered=True)
            email_norm = email.strip().lower()
            hashed_pw = hash_password(password)

            table = "doctors" if user_type == "Doctor" else "patients"

            # Explicitly select all columns; many schemas include 'id' but some may use 'patient_id'
            cursor.execute(
                f"SELECT * FROM {table} WHERE email=%s AND password=%s LIMIT 1",
                (email_norm, hashed_pw)
            )
            row = cursor.fetchone()
            return row

        except Error as e:
            st.error(f"Database error during login: {e}")
            return None

        finally:
            try:
                cursor.close()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass

# ----------------------------
# Session state and navigation
# ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_type = None
    st.session_state.user = None
    st.session_state.current_page = None  # 'patient', 'doctor', 'checkup', etc.
    # Splash screen shown once before login/signup
    st.session_state.splash_shown = False
    # Ensure history table exists when app starts
    ensure_history_table()
    # Ensure appointment_time column exists
    ensure_appointment_time_column()
    # Ensure urgency column exists (CS 213)
    ensure_urgency_column()

def navigate_to(page_name: str):
    st.session_state.current_page = page_name
    safe_rerun()

# ----------------------------
# Helper: resolve patient_id
# ----------------------------
def resolve_patient_id(user):
    """
    Attempt to resolve the patient's primary key (patient_id) for inserts.
    Strategy:
      1. Check common keys in the provided user dict: 'id', 'ID', 'patient_id', 'patientId'
      2. Fallback to a DB lookup by the user's email (patients.email)
    Returns the patient id (int or string) or None if not found.
    """
    if not user:
        return None

    # If the user is not a dict (e.g., tuple from cursor), attempt to normalize
    # into a dict so `.get()` works below. This helps when cursor wasn't
    # configured with `dictionary=True`.
    if not isinstance(user, dict):
        try:
            # Try to extract common attributes; if not present, we'll fallback
            normalized = {}
            for attr in ("id", "ID", "patient_id", "patientId", "email", "firstname", "middlename", "lastname", "contact"):
                try:
                    normalized[attr] = getattr(user, attr)
                except Exception:
                    try:
                        # If user is a sequence/tuple and attr corresponds to index, skip
                        pass
                    except Exception:
                        pass
            # remove keys with None values to keep behavior consistent
            user = {k: v for k, v in normalized.items() if v is not None} or {}
        except Exception:
            user = {}

    # 1) Common keys (check both lowercase and uppercase)
    for key in ("id", "ID", "patient_id", "patientId"):
        pid = None
        try:
            pid = user.get(key) if isinstance(user, dict) else None
        except Exception:
            pid = None
        if pid:
            return pid

    # 2) Try DB lookup by email
    email = user.get("email")
    if not email:
        return None

    conn = get_connection()
    if not conn:
        return None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id FROM patients WHERE email=%s LIMIT 1", (email.strip().lower(),))
        row = cursor.fetchone()
        if row:
            return row.get("id")
        return None
    except Error:
        return None
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
def save_doctor_availability(doctor_id, avail_start, avail_end):
    """Save doctor's availability to database."""
    conn = get_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS doctor_availability (
                doctor_id INT PRIMARY KEY,
                avail_start TIME,
                avail_end TIME,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            INSERT INTO doctor_availability (doctor_id, avail_start, avail_end, updated_at)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
            ON DUPLICATE KEY UPDATE
            avail_start=VALUES(avail_start), avail_end=VALUES(avail_end), updated_at=CURRENT_TIMESTAMP
        """, (doctor_id, avail_start, avail_end))
        conn.commit()
        return True
    except Exception as e:
        # Log the error for debugging
        return False
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

def get_doctor_availability(doctor_id=None):
    """Fetch a doctor's availability from database.

    If `doctor_id` is provided, fetch availability for that doctor.
    Otherwise return the most recently updated availability (for patients booking).
    Returns (avail_start, avail_end) or (None, None).
    """
    conn = get_connection()
    if not conn:
        return None, None
    try:
        cursor = conn.cursor(dictionary=True)
        if doctor_id is not None:
            cursor.execute(
                """
                SELECT avail_start, avail_end FROM doctor_availability
                WHERE doctor_id = %s LIMIT 1
                """,
                (doctor_id,)
            )
        else:
            # Fetch most recently updated availability for patients
            cursor.execute(
                """
                SELECT avail_start, avail_end FROM doctor_availability
                ORDER BY updated_at DESC LIMIT 1
                """
            )
        row = cursor.fetchone()
        if row:
            return row.get("avail_start"), row.get("avail_end")
        return None, None
    except Exception as e:
        return None, None
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

    # (moved into the function body earlier) -- duplicate lookup removed

# Archive/finish a checkup: move the row into a history table and remove from queue.
def finish_checkup(checkup_id):
    # Ensure history table exists first
    ensure_history_table()
    
    conn = get_connection()
    if not conn:
        st.error("Could not connect to DB to finish checkup.")
        return False

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM checkups WHERE id=%s LIMIT 1", (checkup_id,))
        row = cursor.fetchone()
        if not row:
            st.warning("Checkup not found or already processed.")
            return False

        finished_at = datetime.now(timezone.utc)
        
        inserted = False
        error_msg = None
        
        # Get the actual columns in checkups_history table
        try:
            cursor.execute("DESCRIBE checkups_history")
            describe_result = cursor.fetchall()
            history_cols = set()
            for col_info in describe_result:
                # When using dictionary=True, col_info is a dict with 'Field', 'Type', etc.
                if isinstance(col_info, dict):
                    history_cols.add(col_info.get('Field'))
                else:
                    # Fallback for tuple format
                    history_cols.add(col_info[0])
        except Exception as e:
            st.error(f"Could not get checkups_history schema: {e}")
            return False
        
        # Build list of columns that exist in both the row data and the history table
        cols_to_insert = []
        values_to_insert = []
        
        # Include 'id' from the original checkup (don't let it auto-increment)
        if 'id' in history_cols:
            cols_to_insert.append('id')
            values_to_insert.append(row.get('id'))
        
        # Add all other columns
        for col in row.keys():
            if col != 'id' and col in history_cols:
                cols_to_insert.append(col)
                values_to_insert.append(row.get(col))
        
        # Always add finished_at
        if 'finished_at' in history_cols:
            cols_to_insert.append('finished_at')
            values_to_insert.append(finished_at)
        
        # Try the insert with only matching columns
        if cols_to_insert:
            try:
                placeholders = ", ".join(["%s"] * len(cols_to_insert))
                col_names = ", ".join(cols_to_insert)
                
                cursor.execute(f"INSERT INTO checkups_history ({col_names}) VALUES ({placeholders})", tuple(values_to_insert))
                conn.commit()
                inserted = True
            except Exception as e:
                error_msg = str(e)
        else:
            error_msg = "No matching columns found between checkup and history table"

        if inserted:
            # Successfully archived, now delete from queue
            try:
                cursor.execute("DELETE FROM checkups WHERE id=%s", (checkup_id,))
                conn.commit()
                return True
            except Exception as e:
                st.error(f"Archived checkup but failed to remove from queue: {e}")
                return False
        else:
            # Couldn't archive - show specific error and don't delete from queue
            st.error(f"Could not archive checkup to history tables. Error: {error_msg}")
            st.error("The checkup remains in the queue. Please ensure the checkups_history table exists and has the correct schema.")
            return False

    except Exception as e:
        st.error(f"Unexpected error in finish_checkup: {e}")
        return False
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

# ----------------------------
# Pages
# ----------------------------

def render_app_header():
    """Render a persistent app header with logo and title."""
    logo_path = r"C:\Users\bobon\Desktop\Mediqueue pictures\Mediqueue no label.jpg"
    logo = logo_path

    header_col1, header_col2 = st.columns([1, 6])
    with header_col1:
        if logo:
            try:
                st.image(logo, width=80)
            except Exception:
                pass
        with header_col2:
            st.markdown("""
            <div style='padding: 10px 0;'>
                <h1 style='margin:0; color:#0f172a'>MediQueue</h1>
                <div style='color:#6b7280'>Clinic Queue Management</div>
            </div>
            """, unsafe_allow_html=True)

def render_sidebar_profile(user: dict, role_label: str = "Patient"):
    """Render a compact profile card in the sidebar for given user and role.

    This shows the avatar, name, role, email and contact (if present) and
    provides an inline edit form that updates `st.session_state.user`.
    """
    first = (user.get("firstname") or "").strip()
    middle = (user.get("middlename") or "").strip()
    last = (user.get("lastname") or "").strip()
    full_name = " ".join([n for n in [first, middle, last] if n]).strip() or role_label

    user_pic = user.get("picture") or "https://via.placeholder.com/150"
    st.sidebar.image(user_pic, width=120)
    display_name = f"Dr. {full_name}" if role_label == "Doctor" else full_name
    st.sidebar.markdown(f"**{display_name}**")


def render_profile_main(user: dict, role_label: str = "Patient"):
    """Render profile details and edit form in the main page area."""
    first = (user.get("firstname") or "").strip()
    middle = (user.get("middlename") or "").strip()
    last = (user.get("lastname") or "").strip()
    full_name = " ".join([n for n in [first, middle, last] if n]).strip() or role_label

    st.markdown("**Profile**")
    cols = st.columns([1, 3])
    with cols[0]:
        st.image(user.get("picture") or "https://via.placeholder.com/150", width=120)
    with cols[1]:
        st.write(f"**Name:** {full_name}")
        st.write(f"**Role:** {role_label}")
        if user.get("email"):
            st.write(f"**Email:** {user.get('email')}")
        if user.get("contact"):
            st.write(f"**Contact:** {user.get('contact')}")

    if st.button("Edit Profile", key=f"edit_main_{role_label}"):
        st.session_state.edit_profile = True

    if st.session_state.get("edit_profile"):
        with st.form("edit_profile_main_form"):
            new_first = st.text_input("First name", value=first)
            new_middle = st.text_input("Middle name", value=middle)
            new_last = st.text_input("Last name", value=last)
            new_email = st.text_input("Email", value=user.get("email", ""))
            new_contact = st.text_input("Contact", value=user.get("contact", ""))
            submitted = st.form_submit_button("Save")
            if submitted:
                usr = st.session_state.user or {}
                usr.update({
                    "firstname": new_first,
                    "middlename": new_middle,
                    "lastname": new_last,
                    "email": new_email,
                    "contact": new_contact,
                })
                st.session_state.user = usr
                st.session_state.edit_profile = False
                try:
                    st.experimental_rerun()
                except Exception:
                    safe_rerun()


def blank_patient_page():
    user = st.session_state.user or {}

    # Match doctor's sidebar style for consistency
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                background-color: #00bcd4; /* aqua-blue */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Optional logo in the sidebar
    try:
        st.sidebar.image(r"C:\Users\bobon\Desktop\Mediqueue pictures\Mediqueue no label.jpg", width=150)
    except Exception:
        pass

    first = user.get("firstname") or ""
    middle = user.get("middlename") or ""
    last = user.get("lastname") or ""
    full_name = " ".join([n for n in [first, middle, last] if n]).strip() or "Patient"

    # render compact profile card
    try:
        render_sidebar_profile(user, role_label="Patient")
    except Exception:
        # fallback to original simple display
        user_pic = user.get("picture") or "https://via.placeholder.com/150"
        st.sidebar.image(user_pic, width=150)
        st.sidebar.markdown(f"**{full_name}**")

    menu = st.sidebar.radio("Menu", ["Profile", "Home", "History", "Logout"]) 

    if menu == "Profile":
        st.subheader("My Profile")
        # Show full profile in main area
        render_profile_main(user, role_label="Patient")
    elif menu == "Home":
        st.subheader(f"Welcome {full_name}")
        st.markdown("---")
        # --- Begin moved: My Checkups section (moved into Home) ---
        patient_id = resolve_patient_id(user)
        if not patient_id:
            st.error("Unable to retrieve your patient ID. Please log in again.")
        else:
            # Ensure appointment_date column exists
            ensure_appointment_date_column()
            
            checkups = get_patient_checkups(patient_id)
            
            # Check for upcoming appointments within 60 minutes
            upcoming_soon = []
            if checkups:
                from datetime import datetime
                now = datetime.now()
                for checkup in checkups:
                    appt_date = checkup.get("appointment_date")
                    appt_time = checkup.get("appointment_time")
                    if appt_date and appt_time:
                        try:
                            # Parse appointment datetime
                            appt_datetime_str = f"{appt_date} {appt_time}"
                            appt_datetime = datetime.strptime(appt_datetime_str, "%Y-%m-%d %H:%M")
                            
                            # Calculate minutes until appointment
                            time_diff = (appt_datetime - now).total_seconds() / 60
                            
                            # If appointment is within 60 mins and hasn't passed
                            if 0 < time_diff <= 60:
                                upcoming_soon.append({
                                    "checkup": checkup,
                                    "minutes_left": time_diff
                                })
                        except Exception:
                            pass
            
            # Display notifications for upcoming appointments
            if upcoming_soon:
                for item in upcoming_soon:
                    checkup = item["checkup"]
                    mins = item["minutes_left"]
                    doctor_name = f"Dr. {checkup.get('firstname', '')} {checkup.get('lastname', '')}" if checkup.get("firstname") else "Your Doctor"
                    st.warning(f"⏰ **Upcoming Appointment Alert!** Your checkup with {doctor_name} is in {mins:.0f} minutes! ({checkup.get('appointment_time', 'N/A')})")

            
            if not checkups:
                st.info("📋 You have no scheduled checkups yet. Click 'Add Check Up' on the Home page to schedule one.")
            else:
                st.write(f"You have **{len(checkups)}** scheduled checkup(s):")
                st.markdown("---")
                
                for idx, checkup in enumerate(checkups, 1):
                    with st.container():
                        # Create columns for layout
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        # Display checkup info
                        with col1:
                            doctor_name = "TBD"
                            if checkup.get("firstname") and checkup.get("lastname"):
                                doctor_name = f"Dr. {checkup['firstname']} {checkup['lastname']}"
                            
                            specialty = checkup.get("specialty") or "General"
                            urgency_level = checkup.get("urgency") or 1
                            urgency_text = ["", "Low", "Medium", "High", "Critical"][min(int(urgency_level), 4)]
                            
                            # Appointment datetime
                            appt_date = checkup.get("appointment_date") or "TBD"
                            appt_time = checkup.get("appointment_time") or "TBD"
                            
                            st.markdown(f"""
                            **Checkup #{idx}**
                            - **Doctor:** {doctor_name} ({specialty})
                            - **Appointment:** {appt_date} at {appt_time}
                            - **Urgency:** {urgency_text}
                            - **Symptoms:** {checkup.get('symptoms', 'Not specified')}
                            """)
                        
                        # View button
                        with col2:
                            if st.button("👁️ View", key=f"view_{checkup['id']}"):
                                st.session_state[f"view_checkup_{checkup['id']}"] = True
                        
                        # Cancel button (allowed during first 12 hours after check-in)
                        with col3:
                            can_cancel, hours_left, cancel_msg = can_cancel_from_checkin(
                                checkup.get("created_at")
                            )

                            if can_cancel:
                                # Show remaining time and cancel button while within 12-hour window
                                st.caption(f"⏳ {hours_left:.1f} hours remaining to cancel")
                                if st.button("❌ Cancel", key=f"cancel_{checkup['id']}"):
                                    success, message = cancel_checkup(checkup['id'], patient_id)
                                    if success:
                                        st.success(message)
                                        st.session_state[f"rerun_{idx}"] = True
                                        try:
                                            st.experimental_rerun()
                                        except Exception:
                                            safe_rerun()
                                    else:
                                        st.error(message)
                            else:
                                st.caption(f"✅ Cancellation window closed — {cancel_msg}")
                        
                        # Show detailed view if requested
                        if st.session_state.get(f"view_checkup_{checkup['id']}"):
                            st.markdown("---")
                            st.write("**Full Details:**")
                            st.json({
                                "ID": checkup['id'],
                                "Patient Name": checkup['fullname'],
                                "Age": checkup['age'],
                                "Contact": checkup['contact'],
                                "Symptoms": checkup['symptoms'],
                                "Doctor": doctor_name,
                                "Appointment Date": appt_date,
                                "Appointment Time": appt_time,
                                "Urgency": urgency_text,
                                "Created At": checkup['created_at']
                            })
                            if st.button("Close Details", key=f"close_{checkup['id']}"):
                                st.session_state[f"view_checkup_{checkup['id']}"] = False
                                try:
                                    st.experimental_rerun()
                                except Exception:
                                    safe_rerun()
                        
                        st.markdown("---")
        # --- End moved: My Checkups section ---
        st.write("Quick actions:")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Add Check Up"):
                # Directly set the target page and force a rerun so one click navigates immediately
                st.session_state.current_page = "checkup"
                try:
                    st.experimental_rerun()
                except Exception:
                    # Fallback to the safe helper
                    safe_rerun()
        with col2:
            st.write("Click 'Add Check Up' to open the check up form.")

        st.markdown("---")

    elif menu == "History":
        st.subheader("Your History")
        st.write("Past checkups and visits will appear here.")
        # Attempt to show patient history if available
        try:
            patient_id = resolve_patient_id(user)
            conn = get_connection()
            # Debug helper: show resolved patient/session info for troubleshooting
            if SHOW_DEBUG:
                with st.expander("Debug: Resolved patient/session info (click to view)"):
                    st.write({
                        "resolved_patient_id": patient_id,
                        "user_keys": list(user.keys()) if isinstance(user, dict) else str(type(user)),
                        "user_obj": user,
                    })
                    # Show some sample history rows unfiltered to inspect DB contents
                    sample_conn = get_connection()
                    if sample_conn:
                        try:
                            sample_cursor = sample_conn.cursor(dictionary=True)
                            sample_cursor.execute("SELECT id, patient_id, fullname, finished_at FROM checkups_history ORDER BY finished_at DESC LIMIT 10")
                            sample_rows = sample_cursor.fetchall()
                            st.write({"sample_checkups_history": sample_rows})
                        except Exception as e:
                            st.write({"sample_checkups_history_error": str(e)})
                        finally:
                            try:
                                sample_cursor.close()
                            except Exception:
                                pass
                            try:
                                sample_conn.close()
                            except Exception:
                                pass
            if conn and patient_id:
                cursor = conn.cursor(dictionary=True)
                rows = []
                
                # Try to fetch from checkups_history first
                try:
                    cursor.execute(
                        "SELECT id, fullname, created_at, finished_at, symptoms FROM checkups_history WHERE patient_id=%s ORDER BY finished_at DESC",
                        (patient_id,)
                    )
                    rows = cursor.fetchall()
                except Exception as e1:
                    # If that fails, try without the id column (in case it doesn't exist in history table)
                    try:
                        cursor.execute(
                            "SELECT fullname, created_at, finished_at, symptoms FROM checkups_history WHERE patient_id=%s ORDER BY finished_at DESC",
                            (patient_id,)
                        )
                        rows = cursor.fetchall()
                    except Exception as e2:
                        # Try history table as fallback
                        try:
                            cursor.execute(
                                "SELECT fullname, created_at, finished_at, symptoms FROM history WHERE patient_id=%s ORDER BY finished_at DESC",
                                (patient_id,)
                            )
                            rows = cursor.fetchall()
                        except Exception as e3:
                            st.warning(f"Could not fetch history: {e3}")
                
                if rows:
                    st.table(rows)
                else:
                    st.info("No history found.")
                try:
                    cursor.close()
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
            else:
                st.info("No history available for this account.")
        except Exception:
            st.info("Unable to load history at this time.")

    elif menu == "Logout":
        st.session_state.logged_in = False
        st.session_state.user = None
        st.session_state.user_type = None
        safe_rerun()
    return

def blank_doctor_page():
    user = st.session_state.user or {}

    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                background-color: #00bcd4; /* aqua-blue */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Adjust the logo path as needed
    try:
        st.sidebar.image(r"C:\Users\bobon\Desktop\Mediqueue pictures\Mediqueue no label.jpg", width=150)
    except Exception:
        pass

    first = user.get("firstname") or ""
    middle = user.get("middlename") or ""
    last = user.get("lastname") or ""
    full_name = " ".join([n for n in [first, middle, last] if n]).strip() or "Doctor"

    # render compact profile card for doctor
    try:
        render_sidebar_profile(user, role_label="Doctor")
    except Exception:
        user_pic = user.get("picture") or "https://via.placeholder.com/150"
        st.sidebar.image(user_pic, width=150)
        st.sidebar.markdown(f"**Dr. {full_name}**")

    menu = st.sidebar.radio("Menu", ["Profile", "Home", "Queue", "History", "Logout"])

    if menu == "Profile":
        st.subheader("My Profile")
        render_profile_main(user, role_label="Doctor")
    elif menu == "Home":
        st.subheader(f"Welcome Dr. {full_name}")
        st.write("Overview and quick actions for the doctor.")
        
        # Fetch completed checkups for progress bar
        conn = get_connection()
        history_checkups = []
        if conn:
            try:
                cursor = conn.cursor(dictionary=True)
                try:
                    cursor.execute(
                        "SELECT id, patient_id, fullname, created_at, finished_at, symptoms FROM checkups_history ORDER BY finished_at DESC"
                    )
                    history_checkups = cursor.fetchall()
                except Exception:
                    try:
                        cursor.execute(
                            "SELECT id, patient_id, fullname, created_at, finished_at, symptoms FROM history ORDER BY finished_at DESC"
                        )
                        history_checkups = cursor.fetchall()
                    except Exception:
                        pass
            except Error:
                pass
            finally:
                try:
                    cursor.close()
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
        
        completed_checkups = len(history_checkups)
        
        # Get total checkups (both queue and history) for completion percentage
        total_conn = get_connection()
        if total_conn:
            try:
                total_cursor = total_conn.cursor(dictionary=True)
                try:
                    total_cursor.execute("SELECT COUNT(*) as count FROM checkups")
                    queue_result = total_cursor.fetchone()
                    queue_count = queue_result.get("count", 0) if queue_result else 0
                except Exception:
                    queue_count = 0
                total_checkups = queue_count + completed_checkups
            except Exception:
                total_checkups = completed_checkups
            finally:
                try:
                    total_cursor.close()
                except Exception:
                    pass
                try:
                    total_conn.close()
                except Exception:
                    pass
        else:
            total_checkups = completed_checkups
        
        # Calculate percentage
        if total_checkups > 0:
            completion_percentage = (completed_checkups / total_checkups) * 100
        else:
            completion_percentage = 0
        
        # Display circular progress bar
        st.markdown("---")
        st.subheader("Your Progress Today")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.markdown(f"""
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
                <svg width="150" height="150" style="transform: rotate(-90deg);">
                    <circle cx="75" cy="75" r="70" fill="none" stroke="#e0e0e0" stroke-width="8" />
                    <circle cx="75" cy="75" r="70" fill="none" stroke="#4CAF50" stroke-width="8" 
                            stroke-dasharray="{completion_percentage * 4.4:.1f}, 440" stroke-linecap="round" />
                </svg>
                <div style="text-align: center; margin-top: -100px;">
                    <div style="font-size: 24px; font-weight: bold; color: #4CAF50;">
                        {completion_percentage:.1f}%
                    </div>
                    <div style="font-size: 12px; color: #677;">
                        {completed_checkups} of {total_checkups} completed
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("")
        st.markdown("")
        st.markdown("---")
        
        # CS 213: Display Queue Statistics : Statistical Analysis & Mathematical Modeling)- moved to Home
        st.subheader("Queue Analytics")
        # Fetch pending checkups for analytics
        analytics_conn = get_connection()
        analytics_checkups = []
        if analytics_conn:
            try:
                analytics_cursor = analytics_conn.cursor(dictionary=True)
                doctor_id = user.get("ID") or user.get("id") or user.get("doctor_id")
                if doctor_id:
                    try:
                        sql = "SELECT id, patient_id, fullname, created_at, appointment_time, symptoms, urgency FROM checkups WHERE doctor_id=%s ORDER BY urgency DESC, created_at ASC"
                        analytics_cursor.execute(sql, (doctor_id,))
                        analytics_checkups = analytics_cursor.fetchall()
                        analytics_checkups = sort_checkups(analytics_checkups, method="avl", key="urgency", reverse=True)
                    except Error:
                        try:
                            sql = "SELECT id, patient_id, fullname, created_at, appointment_time, symptoms FROM checkups WHERE doctor_id=%s ORDER BY created_at ASC"
                            analytics_cursor.execute(sql, (doctor_id,))
                            analytics_checkups = analytics_cursor.fetchall()
                            for c in analytics_checkups:
                                if "urgency" not in c:
                                    c["urgency"] = 2
                            analytics_checkups = sort_checkups(analytics_checkups, method="avl", key="urgency", reverse=True)
                        except Error:
                            pass
            except Error:
                pass
            finally:
                try:
                    analytics_cursor.close()
                except Exception:
                    pass
                try:
                    analytics_conn.close()
                except Exception:
                    pass
        
        if analytics_checkups:
            # Calculate queue statistics
            queue_stats = calculate_queue_statistics(analytics_checkups)
            
            # Display key metrics in columns
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                st.metric("Patients in Queue", queue_stats["total_patients"])
            
            with col_stat2:
                avg_wait = queue_stats["average_wait_time"]
                st.metric("Average Wait Time", f"{avg_wait} min")
            
            with col_stat3:
                max_wait = queue_stats["max_wait_time"]
                st.metric("Last Patient Wait", f"{max_wait} min")
            
            with col_stat4:
                service_level = calculate_service_level(analytics_checkups, max_acceptable_wait=60)
                st.metric("Service Level (≤60 min)", f"{service_level}%")
            
            # Display urgency distribution
            col_urg1, col_urg2 = st.columns(2)
            with col_urg1:
                st.markdown("**Urgency Distribution**")
                urg_dist = queue_stats["urgency_distribution"]
                urg_data = f"""
                🟢 Low: {urg_dist[1]} patients
                🟡 Medium: {urg_dist[2]} patients
                🟠 High: {urg_dist[3]} patients
                🔴 Critical: {urg_dist[4]} patients
                """
                st.info(urg_data)
            
            with col_urg2:
                st.markdown("**Next Patient (Greedy Algorithm)**")
                next_patient = select_next_patient_greedy(analytics_checkups)
                if next_patient:
                    next_name = next_patient.get("fullname", "Unknown")
                    next_urg = next_patient.get("urgency", 2)
                    next_urg_name = URGENCY_ID_TO_NAME.get(next_urg, "Unknown")
                    st.success(f"👤 {next_name}\n\n⚠️ {next_urg_name}")
        
        st.markdown("---")
        
        # CS 213: Department Flow Manager (array/linked/heap views) - moved to Home
        st.subheader("Department Queues")
        # Fetch pending checkups for department queue building
        home_conn = get_connection()
        home_checkups = []
        if home_conn:
            try:
                home_cursor = home_conn.cursor(dictionary=True)
                doctor_id = user.get("ID") or user.get("id") or user.get("doctor_id")
                if doctor_id:
                    try:
                        sql = "SELECT id, patient_id, fullname, created_at, appointment_time, symptoms, urgency FROM checkups WHERE doctor_id=%s ORDER BY urgency DESC, created_at ASC"
                        home_cursor.execute(sql, (doctor_id,))
                        home_checkups = home_cursor.fetchall()
                        home_checkups = sort_checkups(home_checkups, method="avl", key="urgency", reverse=True)
                    except Error:
                        try:
                            sql = "SELECT id, patient_id, fullname, created_at, appointment_time, symptoms FROM checkups WHERE doctor_id=%s ORDER BY created_at ASC"
                            home_cursor.execute(sql, (doctor_id,))
                            home_checkups = home_cursor.fetchall()
                            for c in home_checkups:
                                if "urgency" not in c:
                                    c["urgency"] = 2
                            home_checkups = sort_checkups(home_checkups, method="avl", key="urgency", reverse=True)
                        except Error:
                            pass
            except Error:
                pass
            finally:
                try:
                    home_cursor.close()
                except Exception:
                    pass
                try:
                    home_conn.close()
                except Exception:
                    pass
        
        if home_checkups:
            dq = DepartmentQueues()
            dq.build_from_checkups(home_checkups)
            depts = list(dq.queues.keys()) or ["General"]

            col_d1, col_d2 = st.columns([2, 1])
            with col_d1:
                selected_dept = st.selectbox("Select department:", depts, key="home_dept_select")
            with col_d2:
                view_method = st.selectbox("View as:", ["array", "linked", "heap"], key="home_view_method")

            snapshot = dq.get_queue_snapshot(selected_dept, method=view_method)
            st.markdown(f"**{selected_dept} queue ({view_method}) — {len(snapshot)} patients**")
            if snapshot:
                for i, p in enumerate(snapshot, start=1):
                    pname = p.get("fullname") or p.get("patient_id") or "Unknown"
                    purg = URGENCY_ID_TO_NAME.get(p.get("urgency", 2), "Unknown")
                    st.write(f"{i}. {pname} — {purg}")
            else:
                st.info("No patients in this department queue.")
        else:
            st.info("No checkups available to display department queues.")
        
        # CS 213: Queue Overflow Prediction (Poisson Distribution) - moved to Home
        if home_checkups:
            st.markdown("---")
            st.subheader("Queue Overflow Prediction")
            
            col_pred1, col_pred2 = st.columns(2)
            
            with col_pred1:
                # Input: Expected arrival rate (patients per hour)
                arrival_rate = st.slider(
                    "Expected patient arrivals per hour:",
                    min_value=1,
                    max_value=20,
                    value=5,
                    step=1,
                    key="home_arrival_rate_slider"
                )
            
            with col_pred2:
                # Input: Queue capacity
                max_capacity = st.slider(
                    "Maximum queue capacity:",
                    min_value=20,
                    max_value=200,
                    value=100,
                    step=10,
                    key="home_max_capacity_slider"
                )
            
            # Calculate overflow prediction
            overflow_pred = predict_queue_overflow(
                current_queue_size=len(home_checkups),
                arrival_rate=arrival_rate,
                max_capacity=max_capacity
            )
            
            # Display predictions
            col_p1, col_p2, col_p3 = st.columns(3)
            
            with col_p1:
                expected_arr = overflow_pred["expected_arrivals"]
                st.metric("Expected Arrivals (1hr)", f"{expected_arr} patients")
            
            with col_p2:
                proj_queue = overflow_pred["projected_queue_size"]
                st.metric("Projected Queue Size", f"{proj_queue:.0f} patients")
            
            with col_p3:
                overflow_prob = overflow_pred["overflow_probability"]
                if overflow_prob > 50:
                    st.error(f"⚠️ Overflow Risk: {overflow_prob}%")
                elif overflow_prob > 25:
                    st.warning(f"⚡ Overflow Risk: {overflow_prob}%")
                else:
                    st.success(f"✓ Overflow Risk: {overflow_prob}%")
            
            # Show if will overflow
            if overflow_pred["will_overflow"]:
                st.warning(f"🚨 Queue will likely overflow! Consider staffing additional doctors.")
            else:
                st.info(f"✅ Queue should remain manageable.")
            
            # CS 213: Advanced Discrete Math Analytics - moved to Home
            st.markdown("---")
            st.subheader("Advanced Analytics ")
            
            # Calculate queue statistics for Home display
            home_queue_stats = calculate_queue_statistics(home_checkups)
            
            cancel_col1, cancel_col2 = st.columns(2)
            with cancel_col1:
                st.markdown("**Cancellation Risk**")
                cancel_prob = forecast_cancellation_probability(arrival_rate, 0.05)
                st.metric(f"Chance: {cancel_prob}%", "Discrete Probability Model")
            with cancel_col2:
                st.markdown("**Late Arrivals**")
                late_prob = forecast_late_arrivals(arrival_rate, 0.10)
                st.metric(f"Chance: {late_prob}%", "Forecast Model")
            
            crowd_col1, crowd_col2 = st.columns(2)
            with crowd_col1:
                st.markdown("**Physical Crowding**")
                crowding = calculate_crowding_index(len(home_checkups), 30, 100)
                st.metric(f"Index: {crowding}%", "Space Assessment")
            with crowd_col2:
                st.markdown("**Patient Satisfaction**")
                avg_wait = home_queue_stats.get("average_wait_time", 30)
                avg_urg = int(sum(c.get("urgency", 2) for c in home_checkups) / max(len(home_checkups), 1)) if home_checkups else 2
                satisfaction = estimate_patient_satisfaction(avg_wait, avg_urg)
                st.metric(f"Score: {satisfaction}/100", "Wait-based Estimate")
            
            st.markdown("---")
            previous = {'avg_wait_time': 45, 'total_patients': 20, 'service_level': 75, 'satisfaction_avg': 65}
            current = {'avg_wait_time': home_queue_stats.get('average_wait_time', 30), 'total_patients': len(home_checkups), 'service_level': calculate_service_level(home_checkups, 60), 'satisfaction_avg': satisfaction}
            improvements = calculate_improvement_metrics(previous, current)
            st.markdown(f"**Improvement Analysis**: {improvements.get('trend', 'N/A')}")
    elif menu == "Queue":
        st.subheader("Doctor Dashboard - Queue")

        # Doctor availability inputs (text) — start and end (HH:MM) (CS 213: Hash Map O(1) Access)
        st.markdown("---")
        st.subheader("Patient Lookup ")
    
        # Build patient hash map for fast lookup
        lookup_conn = get_connection()
        patient_hash_map = {}
        if lookup_conn:
            try:
                lookup_cursor = lookup_conn.cursor(dictionary=True)
                lookup_cursor.execute("SELECT id, firstname, middlename, lastname, age, contact FROM patients LIMIT 1000")
                all_patients_for_lookup = lookup_cursor.fetchall()
                patient_hash_map = build_patient_hash_map(all_patients_for_lookup)
            except Error:
                pass
            finally:
                try:
                    lookup_cursor.close()
                except Exception:
                    pass
                try:
                    lookup_conn.close()
                except Exception:
                    pass

        # Quick patient lookup
        col_lookup1, col_lookup2 = st.columns([3, 1])
        with col_lookup1:
            patient_lookup_id = st.text_input("Enter Patient ID for quick lookup:", key="patient_lookup_input")

        with col_lookup2:
            st.write("")  # Spacing

        if patient_lookup_id:
            # CS 213: O(1) hash map lookup
            found_patient = get_patient_by_id_hash(patient_hash_map, int(patient_lookup_id) if patient_lookup_id.isdigit() else patient_lookup_id)
            if found_patient:
                st.success("✓ Patient Found!")
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    st.metric("Patient ID", found_patient.get("id", "N/A"))
                with col_p2:
                    st.metric("Name", f"{found_patient.get('firstname', '')} {found_patient.get('lastname', '')}".strip())
                with col_p3:
                    st.metric("Age", found_patient.get("age", "N/A"))
                st.info(f"Contact: {found_patient.get('contact', 'N/A')}")
            else:
                st.warning(f"Patient ID {patient_lookup_id} not found in system.")

        st.markdown("---")
        st.subheader("Your Availability")
        # Load doctor's saved availability from database
        doctor_id = user.get("id") or user.get("ID")
        saved_avail_start, saved_avail_end = None, None
        if doctor_id:
            saved_avail_start, saved_avail_end = get_doctor_availability(doctor_id)
        
        col_time1, col_time2 = st.columns([1, 1])
        with col_time1:
            # Start of availability - pre-fill with saved value
            doctor_avail_start = st.text_input(
                "Available from (HH:MM):", 
                value=saved_avail_start or "",
                key="doctor_avail_start"
            )
        with col_time2:
            # End of availability - pre-fill with saved value
            doctor_avail_end = st.text_input(
                "Available until (HH:MM):", 
                value=saved_avail_end or "",
                key="doctor_avail_end"
            )

        # Show current availability if both provided
        if doctor_avail_start and doctor_avail_end:
            st.info(f"Available: {doctor_avail_start} — {doctor_avail_end}")
        elif doctor_avail_start:
            st.info(f"Available from: {doctor_avail_start}")
        elif doctor_avail_end:
            st.info(f"Available until: {doctor_avail_end}")
        else:
            st.info("Set your availability range")
        
        # Confirm button to save availability to database
        if st.button("Confirm Availability"):
            if not doctor_avail_start or not doctor_avail_end:
                st.error("Please enter both start and end times before confirming.")
            else:
                # Validate time format
                try:
                    datetime.strptime(doctor_avail_start.strip(), "%H:%M")
                    datetime.strptime(doctor_avail_end.strip(), "%H:%M")
                    
                    # Save to database
                    doctor_id = user.get("id") or user.get("ID")
                    if doctor_id:
                        result = save_doctor_availability(doctor_id, doctor_avail_start, doctor_avail_end)
                        if result:
                            st.success(f"✅ Availability confirmed: {doctor_avail_start} — {doctor_avail_end}")
                        else:
                            st.error("Failed to save availability. Please try again.")
                    else:
                        st.error("Error: Doctor ID not found. Please log out and log back in.")
                        if SHOW_DEBUG:
                            with st.expander("🔍 Debug Info"):
                                st.write(f"User object: {user}")
                except ValueError:
                    st.error("Invalid time format. Please use HH:MM format (e.g., 09:00)")
        st.markdown("---")

        # Fetch pending checkups from the database - only show checkups assigned to THIS doctor (CS 213: Priority Queue)
        doctor_specialty_display = (user.get("specialty") or user.get("Specialty") or "").strip()
        # Robust doctor id resolution from multiple possible keys or session_state
        doctor_id = (
            user.get("ID") or user.get("id") or user.get("doctor_id") or user.get("user_id") or st.session_state.get("selected_doctor_id")
        )
        # Debug helper: expose resolved id when developing locally
        if SHOW_DEBUG:
            with st.expander("Debug: Resolved doctor/session info (click to view)"):
                st.write({
                    "resolved_doctor_id": doctor_id,
                    "user_keys": list(user.keys()) if isinstance(user, dict) else str(type(user)),
                    "user_obj": user,
                })
        
        conn = get_connection()
        checkups = []
        if conn:
            try:
                cursor = conn.cursor(dictionary=True)
                # Filter by doctor_id to show only checkups assigned to this specific doctor
                try:
                    if doctor_id:
                        sql = (
                            "SELECT id, patient_id, fullname, created_at, appointment_time, symptoms, urgency "
                            "FROM checkups WHERE doctor_id=%s "
                            "ORDER BY urgency DESC, created_at ASC"
                        )
                        cursor.execute(sql, (doctor_id,))
                    else:
                        # If no doctor_id, do not show an intrusive error — inform the user softly
                        st.info("No doctor ID found for this session — queue unavailable. Please re-login if this persists.")
                        checkups = []
                        cursor = None

                    if cursor:
                        checkups = cursor.fetchall()
                        # CS 213: Sort checkups using AVL-based priority ordering (default)
                        checkups = sort_checkups(checkups, method="avl", key="urgency", reverse=True)
                except Error:
                    # Fallback if urgency column doesn't exist yet - try without it
                    try:
                        if doctor_id:
                            sql = (
                                "SELECT id, patient_id, fullname, created_at, appointment_time, symptoms "
                                "FROM checkups WHERE doctor_id=%s "
                                "ORDER BY created_at ASC"
                            )
                            cursor.execute(sql, (doctor_id,))
                            checkups = cursor.fetchall()
                            # Add urgency field with default value
                            for c in checkups:
                                if "urgency" not in c:
                                    c["urgency"] = 2  # Default to Medium
                            # Sort by urgency using AVL as primary structure
                            checkups = sort_checkups(checkups, method="avl", key="urgency", reverse=True)
                    except Error as e:
                        st.error(f"Error fetching queue: {e}")
            except Error as e:
                st.error(f"Error fetching queue: {e}")
            finally:
                try:
                    cursor.close()
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
        if not checkups:
            st.info("No patients in queue.")
            # If queue is empty, help debugging by showing a few checkups without filtering
            if SHOW_DEBUG:
                with st.expander("Debug: sample checkups (unfiltered)"):
                    sample_conn = get_connection()
                    if sample_conn:
                        try:
                            sample_cursor = sample_conn.cursor(dictionary=True)
                            sample_cursor.execute("SELECT id, patient_id, doctor_id, fullname, created_at FROM checkups LIMIT 10")
                            sample_rows = sample_cursor.fetchall()
                            st.write(sample_rows)
                        except Exception as e:
                            st.write(f"Error fetching sample checkups: {e}")
                        finally:
                            try:
                                sample_cursor.close()
                            except Exception:
                                pass
                            try:
                                sample_conn.close()
                            except Exception:
                                pass
        else:
            # Queue list moved to below - analytics and department queues now in Home
            st.markdown("---")
            st.markdown("**Queue list** (sorted by urgency - CS 213: Priority Queue & Merge Sort)")
            # Header row for queue display
            col_h1, col_h2, col_h3, col_h4 = st.columns([1, 2, 2.5, 3])
            col_h1.markdown("**Queue #**")
            col_h2.markdown("**Name**")
            col_h3.markdown("**Time Picked**")
            col_h4.markdown("**Symptoms**")

            for idx, row in enumerate(checkups, start=1):
                row_id = row.get("id") or idx
                name = row.get("fullname") or "Unknown"
                created = row.get("created_at")
                try:
                    if isinstance(created, datetime):
                        time_str = created.strftime("%H:%M")
                    else:
                        time_str = str(created)
                except Exception:
                    time_str = str(created)

                symptoms = row.get("symptoms") or ""
                short = symptoms if len(symptoms) <= 50 else symptoms[:47] + "..."

                # Display main queue row
                col1, col2, col3, col4 = st.columns([1, 2, 2.5, 3])
                col1.write(f"{idx}")
                col2.write(name)
                col3.write(time_str)
                col4.write(short)
                
                # See more button for symptoms
                see_key = f"see_more_{row_id}"
                if col4.button("See more", key=see_key):
                    st.session_state["show_symptoms_id"] = row_id
                    st.session_state["show_symptoms_text"] = symptoms
                
                # Show full symptoms if expanded
                if st.session_state.get("show_symptoms_id") == row_id:
                    st.info(f"Full symptoms: {st.session_state.get('show_symptoms_text', '')}")
                    if st.button("Close", key=f"close_symptoms_{row_id}"):
                        st.session_state.pop("show_symptoms_id", None)
                        st.session_state.pop("show_symptoms_text", None)

                # Action buttons row (Finish and Cancel)
                btn_col1, btn_col2, btn_col3 = st.columns([1.5, 1.5, 3])
                
                finish_key = f"finish_{row_id}"
                if btn_col1.button("✓ Finished", key=finish_key):
                    ok = finish_checkup(row_id)
                    if ok:
                        st.success(f"Checkup {row_id} finished.")
                        safe_rerun()
                    else:
                        st.error("Could not finish checkup.")

                cancel_key = f"cancel_{row_id}"
                if btn_col2.button("✕ Cancel", key=cancel_key):
                    ok = cancel_checkup(row_id, row.get("patient_id"))
                    if ok:
                        st.warning(f"Checkup {row_id} cancelled.")
                        safe_rerun()
                    else:
                        st.error("Could not cancel checkup.")

                st.markdown("---")
            
            st.markdown("---")
    elif menu == "History":
        st.subheader("Doctor Dashboard - History")
        
        # Get the current doctor's ID for filtering
        doctor_id = user.get("id") or user.get("ID")
        
        # Debug: show doctor_id being used
        if SHOW_DEBUG:
            with st.expander("Debug: Doctor ID Info"):
                st.write(f"Doctor ID: {doctor_id}")
                st.write(f"User object: {user}")
        
        # Ensure history table exists with all necessary columns
        ensure_history_table()
        
        # Fetch finished checkups from history for THIS DOCTOR ONLY
        conn = get_connection()
        history_checkups = []
        if conn:
            try:
                cursor = conn.cursor(dictionary=True)
                # Try to fetch from checkups_history without filter first to see if records exist
                try:
                    # Try with doctor_id filter
                    if doctor_id:
                        cursor.execute(
                            "SELECT id, patient_id, fullname, created_at, finished_at, symptoms FROM checkups_history WHERE doctor_id=%s ORDER BY finished_at DESC",
                            (doctor_id,)
                        )
                    else:
                        cursor.execute(
                            "SELECT id, patient_id, fullname, created_at, finished_at, symptoms FROM checkups_history ORDER BY finished_at DESC"
                        )
                    history_checkups = cursor.fetchall()
                    
                    # If no results with doctor_id filter, try without filter to show something
                    if not history_checkups and doctor_id:
                        st.warning(f"No history found for doctor_id={doctor_id}. Showing all history records instead.")
                        cursor.execute(
                            "SELECT id, patient_id, fullname, created_at, finished_at, symptoms FROM checkups_history ORDER BY finished_at DESC"
                        )
                        history_checkups = cursor.fetchall()
                except Exception as e:
                    # If doctor_id filter fails, try without it to get all records
                    st.warning(f"Filtered query failed: {e}. Trying unfiltered query...")
                    try:
                        cursor.execute(
                            "SELECT id, patient_id, fullname, created_at, finished_at, symptoms FROM checkups_history ORDER BY finished_at DESC"
                        )
                        history_checkups = cursor.fetchall()
                        st.warning(f"Retrieved {len(history_checkups)} records (unfiltered)")
                    except Exception as e2:
                        st.error(f"Unfiltered query also failed: {e2}")
            except Error as e:
                st.error(f"Error fetching history: {e}")
            finally:
                try:
                    cursor.close()
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
        
        if not history_checkups:
            st.info("No completed checkups in history.")
            
            # Debug: check if history table has ANY records
            if SHOW_DEBUG:
                with st.expander("Debug: Check history table contents"):
                    debug_conn = get_connection()
                    if debug_conn:
                        try:
                            debug_cursor = debug_conn.cursor(dictionary=True)
                            # Check if table exists and has records
                            try:
                                debug_cursor.execute("SELECT COUNT(*) as cnt FROM checkups_history")
                                count_result = debug_cursor.fetchone()
                                total_records = count_result.get("cnt", 0) if count_result else 0
                                st.write(f"✓ Total records in checkups_history: {total_records}")
                                
                                # Show table columns
                                debug_cursor.execute("DESCRIBE checkups_history")
                                columns = debug_cursor.fetchall()
                                st.write("Columns in checkups_history:")
                                col_names = [c.get("Field") if isinstance(c, dict) else c[0] for c in columns]
                                st.write(col_names)
                                
                                # Check what doctor_id values exist in history
                                st.write("---")
                                st.write("**Doctor IDs in history table:**")
                                debug_cursor.execute("SELECT DISTINCT doctor_id FROM checkups_history")
                                doctor_ids = debug_cursor.fetchall()
                                st.write(f"Unique doctor_ids: {doctor_ids}")
                                
                                # Check records for THIS doctor
                                st.write("---")
                                st.write(f"**Records for doctor_id = {doctor_id}:**")
                                debug_cursor.execute(
                                    "SELECT id, fullname, doctor_id, created_at, finished_at FROM checkups_history WHERE doctor_id=%s LIMIT 5",
                                    (doctor_id,)
                                )
                                matching_records = debug_cursor.fetchall()
                                st.write(f"Found {len(matching_records)} records for this doctor")
                                if matching_records:
                                    st.json(matching_records)
                                
                                if total_records > 0 and len(matching_records) == 0:
                                    # Show first 5 records (all doctors)
                                    st.write("---")
                                    st.write("**First 5 records (any doctor):**")
                                    debug_cursor.execute("SELECT id, fullname, doctor_id, created_at, finished_at FROM checkups_history LIMIT 5")
                                    sample_records = debug_cursor.fetchall()
                                    st.json(sample_records)
                            except Exception as e:
                                st.write(f"Error checking history table: {e}")
                        except Exception as e:
                            st.write(f"Error: {e}")
                        finally:
                            try:
                                debug_cursor.close()
                            except:
                                pass
                            try:
                                debug_conn.close()
                            except:
                                pass
        else:
            st.markdown("**History list**")
            # Header row
            col_h1, col_h2, col_h3, col_h4 = st.columns([2.5, 2, 2, 2])
            col_h1.markdown("**Name**")
            col_h2.markdown("**Time Checked In**")
            col_h3.markdown("**Time Finished**")
            col_h4.markdown("**Actions**")

            for history_row in history_checkups:
                name = history_row.get("fullname") or "Unknown"
                created = history_row.get("created_at")
                finished = history_row.get("finished_at")
                symptoms = history_row.get("symptoms") or ""
                row_id = history_row.get("id")
                
                try:
                    if isinstance(created, datetime):
                        created_str = created.strftime("%H:%M")
                    else:
                        created_str = str(created)
                except Exception:
                    created_str = str(created)
                
                try:
                    if isinstance(finished, datetime):
                        finished_str = finished.strftime("%H:%M")
                    else:
                        finished_str = str(finished)
                except Exception:
                    finished_str = str(finished)

                col1, col2, col3, col4 = st.columns([2.5, 2, 2, 2])
                col1.write(name)
                col2.write(created_str)
                col3.write(finished_str)

                # See more button for symptoms
                see_key = f"history_see_more_{row_id}"
                if col4.button("See more", key=see_key):
                    st.session_state["show_history_symptoms_id"] = row_id
                    st.session_state["show_history_symptoms_text"] = symptoms

                # Show full symptoms if expanded
                if st.session_state.get("show_history_symptoms_id") == row_id:
                    st.info(f"Symptoms: {st.session_state.get('show_history_symptoms_text', '')}")
                    if st.button("Close", key=f"close_history_symptoms_{row_id}"):
                        st.session_state.pop("show_history_symptoms_id", None)
                        st.session_state.pop("show_history_symptoms_text", None)

                st.markdown("---")
    elif menu == "Logout":
        do_logout()

    return

def checkup_form_page():
    st.subheader("Add Check Up")
    user = st.session_state.user or {}

    # Prefill fullname and contact from logged-in patient if available
    first = user.get("firstname") or ""
    middle = user.get("middlename") or ""
    last = user.get("lastname") or ""
    fullname_prefill = " ".join([n for n in [first, middle, last] if n]).strip()

    # Inputs
    fullname = st.text_input("Full name", value=fullname_prefill)
    age = st.text_input("Age", value="")
    contact = st.text_input("Contact", value=user.get("contact", ""))
    symptoms = st.text_area("Symptoms", value="")
    
    # CS 213: Linear Search for Patient ID Validation
    st.markdown("---")
    st.markdown("**Patient Registration Check** (CS 213: Linear Search)")
    
    patient_id = user.get("patient_id") or user.get("id") or None
    
    # Fetch all patients from database for linear search validation
    all_patients = []
    if patient_id:
        conn = get_connection()
        if conn:
            try:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT id, firstname, middlename, lastname FROM patients LIMIT 1000")
                all_patients = cursor.fetchall()
            except Error:
                all_patients = []
            finally:
                try:
                    cursor.close()
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
        
        # CS 213: Use linear search to validate patient ID exists
        existing_patient = linear_search_patient_id(all_patients, patient_id)
        if existing_patient:
            st.success(f"✓ Patient ID {patient_id} verified in system")
        else:
            st.info(f"Note: Patient ID {patient_id} will be registered with this checkup")
    
    st.markdown("---")
    
    # Doctor specialty is now mandatory
    specialty_labels = [name for (_id, name) in SPECIALTIES]
    selected_spec = st.selectbox("Select required doctor specialty:", specialty_labels)
    preferred_specialty_id = SPECIALTY_NAME_TO_ID.get(selected_spec)
    
    # CS 213: Urgency/Priority Level Selection (for priority queue)
    urgency_labels = [name for (_id, name) in URGENCY_LEVELS]
    selected_urgency = st.selectbox("Select urgency level:", urgency_labels)
    selected_urgency_id = URGENCY_NAME_TO_ID.get(selected_urgency)
    
    # Fetch doctors with the selected specialty
    available_doctors = []
    if preferred_specialty_id:
        conn = get_connection()
        if conn:
            try:
                cursor = conn.cursor(dictionary=True)
                # Fetch doctors with matching specialty (numeric id or text match)
                cursor.execute(
                    "SELECT id, firstname, middlename, lastname, specialty FROM doctors WHERE specialty=%s OR specialty=%s",
                    (preferred_specialty_id, selected_spec)
                )
                available_doctors = cursor.fetchall()
            except Error:
                # Fallback if specialty column doesn't exist or query fails
                available_doctors = []
            finally:
                try:
                    cursor.close()
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
    
    # Let patient select a doctor
    selected_doctor = None
    doctor_availability = {}
    if available_doctors:
        doctor_names = [f"{d.get('firstname', '')} {d.get('middlename', '')} {d.get('lastname', '')}".strip() or "Doctor" for d in available_doctors]
        selected_doctor_name = st.selectbox("Select a doctor:", doctor_names)
        selected_doctor_idx = doctor_names.index(selected_doctor_name)
        selected_doctor = available_doctors[selected_doctor_idx]
        
        # Fetch selected doctor's availability
        doctor_id = selected_doctor.get("id")
        if doctor_id:
            saved_avail_start, saved_avail_end = get_doctor_availability(doctor_id)
            if saved_avail_start and saved_avail_end:
                doctor_availability = {"start": saved_avail_start, "end": saved_avail_end}
    else:
        st.warning(f"No doctors available for specialty: {selected_spec}")
    
    # Appointment time input: use selected doctor's availability
    appointment_time = None
    if selected_doctor and doctor_availability:
        db_avail_start = doctor_availability.get("start")
        db_avail_end = doctor_availability.get("end")
    else:
        db_avail_start = None
        db_avail_end = None
    
    avail_start = db_avail_start
    avail_end = db_avail_end

    time_options = None
    if avail_start and avail_end:
        try:
            def to_time(v):
                # Handle timedelta (from MySQL TIME column)
                if isinstance(v, timedelta):
                    total_seconds = int(v.total_seconds())
                    hours = total_seconds // 3600
                    minutes = (total_seconds % 3600) // 60
                    return datetime.min.replace(hour=hours, minute=minutes).time()
                # Handle time/datetime objects
                if hasattr(v, "strftime"):
                    if hasattr(v, 'time'):
                        return v.time()
                    return v
                # Handle strings
                return datetime.strptime(str(v).strip(), "%H:%M").time()

            s_time = to_time(avail_start)
            e_time = to_time(avail_end)
            # Build datetime objects on TOMORROW's date (for future appointments)
            tomorrow = datetime.now().date() + timedelta(days=1)
            cur = datetime.combine(tomorrow, s_time)
            end_dt = datetime.combine(tomorrow, e_time)
            # If end is before start, treat as no available slots
            if end_dt >= cur:
                time_options = []
                while cur <= end_dt:
                    time_options.append(cur.strftime("%H:%M"))
                    cur += timedelta(minutes=30)
        except Exception as e:
            time_options = None

    # Remove times already booked for the selected doctor on the same appointment date
    if time_options and selected_doctor:
        try:
            # appointment_date to check (use same date used above: tomorrow)
            appointment_date = datetime.now().date() + timedelta(days=1)

            def _format_time_val(v):
                try:
                    if v is None:
                        return None
                    if isinstance(v, timedelta):
                        total_seconds = int(v.total_seconds())
                        hours = total_seconds // 3600
                        minutes = (total_seconds % 3600) // 60
                        return f"{hours:02d}:{minutes:02d}"
                    if hasattr(v, "strftime"):
                        # datetime/time-like
                        if hasattr(v, 'time') and not isinstance(v, datetime):
                            return v.strftime("%H:%M")
                        try:
                            return v.time().strftime("%H:%M")
                        except Exception:
                            return v.strftime("%H:%M")
                    s = str(v).strip()
                    for fmt in ("%H:%M:%S", "%H:%M"):
                        try:
                            return datetime.strptime(s, fmt).strftime("%H:%M")
                        except Exception:
                            continue
                    return s
                except Exception:
                    return None

            conn = get_connection()
            if conn:
                try:
                    cursor = conn.cursor(dictionary=True)
                    cursor.execute(
                        "SELECT appointment_time FROM checkups WHERE doctor_id=%s AND appointment_date=%s",
                        (selected_doctor.get("id"), appointment_date)
                    )
                    rows = cursor.fetchall()
                    taken = set()
                    for r in rows:
                        at = None
                        try:
                            if isinstance(r, dict):
                                at = r.get("appointment_time")
                            else:
                                at = r[0]
                        except Exception:
                            at = r
                        fmt = _format_time_val(at)
                        if fmt:
                            taken.add(fmt)

                    # Filter out taken times
                    time_options = [t for t in time_options if t not in taken]
                except Exception:
                    pass
                finally:
                    try:
                        cursor.close()
                    except Exception:
                        pass
                    try:
                        conn.close()
                    except Exception:
                        pass
        except Exception:
            # best-effort only; if anything fails leave time_options as-is
            pass

    # Check if selected doctor has set availability
    if not time_options:
        if selected_doctor:
            st.warning(f"⚠️ Dr. {selected_doctor_name} has not set their availability yet.")
        else:
            st.info("Please select a doctor above to see available appointment times.")
        appointment_time = None
    else:
        if st.session_state.get("confirmed_appointment_time"):
            appointment_time = st.session_state.get("confirmed_appointment_time")
            st.info(f"Confirmed appointment time: {appointment_time}")
        else:
            appointment_time = st.selectbox("Preferred appointment time (30-min intervals):", time_options)

    # Confirm button: locks the appointment time and doctor so they can't be changed afterwards
    if not st.session_state.get("confirmed_appointment_time") and appointment_time is not None and selected_doctor:
        if st.button("Confirm time and doctor"):
            if not appointment_time or not str(appointment_time).strip():
                st.warning("Please choose an appointment time before confirming.")
            elif not selected_doctor:
                st.warning("Please select a doctor before confirming.")
            else:
                try:
                    datetime.strptime(str(appointment_time).strip(), "%H:%M")
                    st.session_state["confirmed_appointment_time"] = str(appointment_time).strip()
                    st.session_state["selected_doctor_id"] = selected_doctor.get("id")
                    st.success(f"Appointment confirmed with {selected_doctor_name} at {st.session_state['confirmed_appointment_time']}")
                    safe_rerun()
                except Exception:
                    st.warning("Invalid time format. Use HH:MM (e.g., 14:30).")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Check Up"):
            # Validate minimal required fields
            if not fullname.strip():
                st.warning("Full name is required.")
                return
            if not contact.strip():
                st.warning("Contact is required.")
                return
            if not selected_doctor:
                st.warning("Please select a doctor before saving.")
                return

            # Use confirmed appointment time if present
            final_appointment = st.session_state.get("confirmed_appointment_time") or appointment_time
            # Validate appointment time format
            if final_appointment and str(final_appointment).strip():
                try:
                    datetime.strptime(str(final_appointment).strip(), "%H:%M")
                except ValueError:
                    st.warning("Invalid appointment time format. Please use HH:MM (e.g., 14:30)")
                    return
            else:
                st.warning("Appointment time is required. Please confirm a time before saving.")
                return

            # Resolve patient_id (ensure it's not null)
            patient_id = resolve_patient_id(user)
            if not patient_id:
                st.error(
                    "Unable to determine patient_id for the logged-in user. "
                    "Make sure the login() returns the user's DB id (commonly 'id' or 'patient_id'), "
                    "or ensure the patient's email exists in patients.email so a lookup can succeed."
                )
                # Helpful debug output (remove in production)
                st.write("Logged user object (for debugging):")
                st.json(user)
                return

            # Try to persist to DB
            conn = get_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    # Convert age if DB column is INT - fallback to None or string
                    if age is None or str(age).strip() == "":
                        age_val = None
                    else:
                        try:
                            age_val = int(age)
                        except ValueError:
                            age_val = age  # store as string if not integer

                    doctor_id = st.session_state.get("selected_doctor_id") or (selected_doctor.get("id") if selected_doctor else None)
                    
                    # Try to include doctor_id and urgency in the insert (CS 213: Priority Queue)
                    appointment_date = (datetime.now() + timedelta(days=1)).date()  # Tomorrow's date
                    # Ensure DB has appointment_date column before inserting (auto-migrate)
                    try:
                        ensure_appointment_date_column()
                    except Exception:
                        # If the migration fails, proceed and let the DB raise a clear error to catch below
                        pass
                    try:
                        cursor.execute(
                            """
                            INSERT INTO checkups
                                (patient_id, fullname, age, contact, symptoms, appointment_date, appointment_time, doctor_id, urgency, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                patient_id,
                                fullname,
                                age_val,
                                contact,
                                symptoms,
                                appointment_date,
                                st.session_state.get("confirmed_appointment_time") or appointment_time,
                                doctor_id,
                                selected_urgency_id,
                                datetime.now(timezone.utc)
                            )
                        )
                    except Error:
                        # Fallback: try without urgency
                        try:
                            cursor.execute(
                                """
                                INSERT INTO checkups
                                    (patient_id, fullname, age, contact, symptoms, appointment_date, appointment_time, doctor_id, created_at)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                                """,
                                (
                                    patient_id,
                                    fullname,
                                    age_val,
                                    contact,
                                    symptoms,
                                    appointment_date,
                                    st.session_state.get("confirmed_appointment_time") or appointment_time,
                                    doctor_id,
                                    datetime.now(timezone.utc)
                                )
                            )
                        except Error:
                            # Final fallback: insert without preferred_specialty
                            cursor.execute(
                                """
                                INSERT INTO checkups
                                    (patient_id, fullname, age, contact, symptoms, appointment_date, appointment_time, created_at)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                """,
                                (
                                    patient_id,
                                    fullname,
                                    age_val,
                                    contact,
                                    symptoms,
                                    appointment_date,
                                    st.session_state.get("confirmed_appointment_time") or appointment_time,
                                    datetime.now(timezone.utc)
                                )
                            )
                    conn.commit()
                    try:
                        saved_id = cursor.lastrowid
                        st.success(f"Check up saved to database (id={saved_id}).")
                    except Exception:
                        st.success("Check up saved to database.")
                    # after save, navigate back to patient dashboard
                    navigate_to("patient")
                except Error as e:
                    st.error(f"Failed to save check up: {e}")
                finally:
                    try:
                        cursor.close()
                    except Exception:
                        pass
                    try:
                        conn.close()
                    except Exception:
                        pass
            else:
                st.error("Could not connect to the database. Check connection settings.")
    with col2:
        if st.button("Cancel"):
            navigate_to("patient")

    return

# ----------------------------
# Logout helper
# ----------------------------
def do_logout():
    st.session_state.logged_in = False
    st.session_state.user_type = None
    st.session_state.user = None
    st.session_state.current_page = None
    safe_rerun()

# ----------------------------
# Main: Login / Signup
# ----------------------------
if not st.session_state.logged_in:

    # Splash screen before showing login/signup
    if not st.session_state.get("splash_shown", False):
        st.markdown("""<div style="width:100vw;margin-left:calc(-50vw + 50%);background:linear-gradient(90deg, #00bcd4 0%, #0097a7 100%);padding:8px 40px;display:flex;justify-content:flex-end;align-items:center;box-shadow:0 2px 8px rgba(0,0,0,0.2);margin-top:-50px;"><div style="display:flex;gap:80px;align-items:center;"><div style="text-align:center;color:white;"><p style="margin:0;font-size:16px;font-weight:600;">About Us</p></div><div style="text-align:center;color:white;"><p style="margin:0;font-size:16px;font-weight:600;">Contact Us</p></div></div></div>""", unsafe_allow_html=True)
        
        st.markdown('<div style="margin-top:-30px;"></div>', unsafe_allow_html=True)
        
        # Allow clicking 'Get started' via URL query param (makes the button actionable)
        params = st.query_params
        # If user clicked the hero link, mark splash as shown and continue (no extra rerun)
        if params.get("getstarted"):
            st.session_state.splash_shown = True
            # clear query params to avoid repeating the action on reload
            try:
                st.query_params = {}
            except Exception:
                pass
        else:
            # Full-viewport hero image with overlayed title (stretches to full browser width)
            st.markdown("""
            <div style="position:relative;width:100vw;left:50%;right:50%;margin-left:-50vw;margin-right:-50vw;height:520px;overflow:hidden;margin-top:-30px;">
                <img src="https://t4.ftcdn.net/jpg/01/83/24/05/360_F_183240566_su0jKCQkOjCXFzUNmNpAVYz6WV4sdBAM.jpg" style="width:100%;height:520px;object-fit:cover;object-position:center;display:block;margin:0;padding:0;"/>
                <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;pointer-events:none;">
                    <h1 style="color:white;font-size:56px;margin:0;text-shadow:0 2px 6px rgba(0,0,0,0.6);">Welcome to MediQueue</h1>
                    <p style="color:rgba(255,255,255,0.95);margin:8px 0 40px 0;font-size:20px;text-shadow:0 1px 4px rgba(0,0,0,0.6);">Your simplified clinic queue and appointment manager</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Get started button overlaid on the hero image — clickable link navigates via query param
            st.markdown("""
            <div style="position:relative;margin-top:-120px;text-align:center;pointer-events:auto;">
                <a href="?getstarted=1" style="background-color:#00bcd4;color:white;border:none;padding:12px 40px;font-size:16px;font-weight:600;border-radius:4px;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,0.2);text-decoration:none;display:inline-block;">Get started</a>
            </div>
            """, unsafe_allow_html=True)
            # Stop here so login/signup form is not rendered while splash is visible
            st.stop()

    try:
        render_app_header()
    except Exception:
        pass

    st.subheader("Sign Up / Login")

    choice = st.radio("Choose an action:", ["Sign Up", "Login"], horizontal=True)

    if choice == "Sign Up":
        st.header("Create an Account")

        user_type = st.selectbox("I am a:", ["Patient", "Doctor"])
        firstname = st.text_input("First Name")
        middlename = st.text_input("Middle Name")
        lastname = st.text_input("Last Name")
        email = st.text_input("Email")
        contact = st.text_input("Contact Number")
        password = st.text_input("Password", type="password")

        # Specialty dropdown for doctors
        specialty = None
        if user_type == "Doctor":
            specialty = st.selectbox("Specialty", [
                "General Practitioner (GP)",
                "Family Medicine",
                "Internal Medicine",
                "Pediatrics",
                "OB-GYN (Obstetrics & Gynecology)",
                "Dermatology",
                "ENT / EENT (Ear, Nose, Throat)",
                "Ophthalmology (Eye Doctor)",
                "Cardiology",
                "Pulmonology",
                "Gastroenterology",
                "Endocrinology (Diabetes/Thyroid)",
                "Psychiatry",
                "Psychology",
                "Urology",
                "Dentistry",
                "Rehabilitation Medicine / Physical Therapy",
                "Allergy & Immunology",
            ])

        if st.button("Sign Up"):
            if firstname and lastname and email and contact and password:
                signup(user_type, firstname, middlename, lastname, email, contact, password, specialty)
            else:
                st.warning("Please fill in all required fields.")

    else:
        st.header("Login to Your Account")

        user_type = st.selectbox("I am a:", ["Patient", "Doctor"])
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if email and password:
                user = login(user_type, email, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user_type = user_type
                    st.session_state.user = user
                    st.session_state.current_page = "doctor" if user_type == "Doctor" else "patient"
                    safe_rerun()
                else:
                    st.error("Invalid email or password.")
            else:
                st.warning("Please enter your email and password.")

# ----------------------------
# Show pages after login
# ----------------------------
else:
    # Render a persistent header so the top banner remains consistent across pages
    try:
        render_app_header()
    except Exception:
        pass

    page = st.session_state.current_page or ("doctor" if st.session_state.user_type == "Doctor" else "patient")
    if page == "doctor":
        blank_doctor_page()
    elif page == "patient":
        blank_patient_page()
    else:
        checkup_form_page()
    

# Advanced Analytics UI Panel for Doctor Dashboard
# This contains the Streamlit code for displaying discrete math metrics

def display_advanced_analytics(checkups, queue_stats, arrival_rate, st):
    """Display advanced discrete mathematics analytics panel"""
    
    # Import functions (assumes they're in main MEDIQUEUE module)
    from MEDIQUEUE import (
        forecast_cancellation_probability,
        forecast_late_arrivals,
        calculate_crowding_index,
        estimate_patient_satisfaction,
        calculate_improvement_metrics,
        calculate_service_level,
        URGENCY_ID_TO_NAME
    )
    
    # CS 213: Advanced Discrete Math Analytics
    st.markdown("---")
    st.subheader("📈 Advanced Analytics (CS 213: Discrete Mathematics & Statistics)")
    
    # Cancellation and late arrival forecasting
    cancel_col1, cancel_col2 = st.columns(2)
    with cancel_col1:
        st.markdown("**Cancellation Risk** (next hour)")
        cancel_prob = forecast_cancellation_probability(arrival_rate, historical_cancellation_rate=0.05)
        if cancel_prob > 60:
            st.error(f"🔴 High Risk: {cancel_prob}% chance")
        elif cancel_prob > 30:
            st.warning(f"🟡 Medium Risk: {cancel_prob}% chance")
        else:
            st.success(f"🟢 Low Risk: {cancel_prob}% chance")
    
    with cancel_col2:
        st.markdown("**Late Arrivals** (next hour)")
        late_prob = forecast_late_arrivals(arrival_rate, historical_late_rate=0.10)
        if late_prob > 60:
            st.error(f"🔴 High Risk: {late_prob}% chance")
        elif late_prob > 30:
            st.warning(f"🟡 Medium Risk: {late_prob}% chance")
        else:
            st.success(f"🟢 Low Risk: {late_prob}% chance")
    
    # Crowding and Satisfaction
    crowd_col1, crowd_col2 = st.columns(2)
    with crowd_col1:
        st.markdown("**Physical Crowding Index**")
        crowding = calculate_crowding_index(len(checkups), max_comfortable_capacity=30, max_safe_capacity=100)
        if crowding >= 80:
            st.error(f"🔴 {crowding}% - Unsafe crowding level")
        elif crowding >= 50:
            st.warning(f"🟡 {crowding}% - Moderate crowding")
        else:
            st.success(f"🟢 {crowding}% - Comfortable")
    
    with crowd_col2:
        st.markdown("**Estimated Patient Satisfaction**")
        avg_wait = queue_stats.get("average_wait_time", 30)
        # Average urgency across queue
        avg_urgency = int(sum(c.get("urgency", 2) for c in checkups) / max(len(checkups), 1)) if checkups else 2
        satisfaction = estimate_patient_satisfaction(avg_wait, urgency_level=avg_urgency)
        if satisfaction >= 70:
            st.success(f"😊 {satisfaction}% satisfaction")
        elif satisfaction >= 40:
            st.warning(f"😐 {satisfaction}% satisfaction")
        else:
            st.error(f"😞 {satisfaction}% satisfaction")
    
    # Improvement tracking (mock data for demo)
    st.markdown("---")
    st.markdown("**Period Comparison & Improvement Analysis**")
    previous = {
        'avg_wait_time': 45,
        'total_patients': 20,
        'service_level': 75,
        'satisfaction_avg': 65
    }
    current = {
        'avg_wait_time': queue_stats.get('average_wait_time', 30),
        'total_patients': len(checkups),
        'service_level': calculate_service_level(checkups, max_acceptable_wait=60),
        'satisfaction_avg': satisfaction
    }
    improvements = calculate_improvement_metrics(previous, current)
    
    imp_c1, imp_c2, imp_c3 = st.columns(3)
    with imp_c1:
        wait_imp = improvements.get('wait_time_improvement_percent', 0)
        if wait_imp > 0:
            st.success(f"⬇️ Wait Time: {wait_imp}% better")
        else:
            st.warning(f"⬆️ Wait Time: {abs(wait_imp)}% worse")
    
    with imp_c2:
        service_imp = improvements.get('service_level_improvement_percent', 0)
        if service_imp > 0:
            st.success(f"📊 Service Level: +{service_imp}%")
        else:
            st.warning(f"📊 Service Level: {service_imp}%")
    
    with imp_c3:
        sat_imp = improvements.get('satisfaction_improvement_percent', 0)
        if sat_imp > 0:
            st.success(f"😊 Satisfaction: +{sat_imp}%")
        else:
            st.warning(f"😊 Satisfaction: {sat_imp}%")
    
    st.info(f"**Trend:** {improvements.get('trend', 'N/A')}")


