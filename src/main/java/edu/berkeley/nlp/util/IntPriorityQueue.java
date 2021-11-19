package edu.berkeley.nlp.util;

import java.text.NumberFormat;
import java.util.Arrays;
import java.util.NoSuchElementException;


/**
 * Max-heap Keys are positions. This implementation maintains a map from
 * (integer) element ids to positions in the heap. Note that these ids must be
 * dense, since the map is backed by a simple array. In particular ids must be
 * in the range [0, maxElem), where maxElem is a parameter passed in at
 * construction time.
 * 
 * @author Adam Pauls
 */
public class IntPriorityQueue
{
	private static final long serialVersionUID = 1L;

	private int size;

	private int capacity;

	private int[] elements;

	private final int[] elementPositions;

	private double[] priorities;

	private final int maxElem;

	/**
	 * 
	 *
	 */
	public IntPriorityQueue(int maxElem) {
		this(maxElem, 15);
	}

	/**
	 * 
	 * @param maxElem
	 *            The id of the largest element in that will ever be added to
	 *            this queue.
	 * @param capacity
	 *            The initial number of entries in the heap. If this number if
	 *            exceeded, the heap will grow dynamically.
	 */
	public IntPriorityQueue(int maxElem, int capacity) {
		this.maxElem = maxElem;
		elementPositions = new int[maxElem + 1];
		Arrays.fill(elementPositions, -1);
		int legalCapacity = 0;
		while (legalCapacity < capacity) {
			legalCapacity = 2 * legalCapacity + 1;
		}
		grow(legalCapacity);
	}

	public boolean hasNext() {
		return !isEmpty();
	}

	/**
	 * Pops the highest scoring element.
	 * 
	 * @return
	 */
	public int next() {
		int first = peek();
		removeFirst();
		return first;
	}

	/**
	 * Returns the highest scoring element without popping it.
	 * 
	 * @return
	 */
	public int peek() {
		if (size() > 0) return elements[0];
		throw new NoSuchElementException();
	}

	/**
	 * Returns the score of the highest-scoring element
	 * 
	 * @return
	 */
	public double getPriorityOfBest() {
		if (size() > 0) return priorities[0];
		return Double.NaN;
	}

	/**
	 * Number of entries currently stored in the heap
	 * 
	 * @return
	 */
	public int size() {
		return size;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.util.PriorityQueueInterface#isEmpty()
	 */
	public boolean isEmpty() {
		return size == 0;
	}

	public boolean put(int key, double priority) {
		if (size == capacity) {
			grow(2 * capacity + 1);
		}
		final int loc = elementPositions[key];
		if (loc >= 0) {
			if (priority > priorities[loc]) increaseKey(key, priority);
			return false;
		} else {
			elements[size] = key;
			elementPositions[key] = size;
			priorities[size] = priority;
			heapifyUp(size);
			size++;
			return true;
		}
	}

	/**
	 * Returns a representation of the queue in decreasing priority order.
	 */
	@Override
	public String toString() {
		return toString(size(), false);
	}

	/**
	 * Returns a representation of the queue in decreasing priority order,
	 * displaying at most maxKeysToPrint elements and optionally printing one
	 * element per line.
	 * 
	 * @param maxKeysToPrint
	 * @param multiline
	 *            TODO
	 */
	public String toString(int maxKeysToPrint, boolean multiline) {
		IntPriorityQueue pq = clone();
		StringBuilder sb = new StringBuilder(multiline ? "" : "[");
		int numKeysPrinted = 0;
		NumberFormat f = NumberFormat.getInstance();
		f.setMaximumFractionDigits(5);
		while (numKeysPrinted < maxKeysToPrint && pq.hasNext()) {
			double priority = pq.getPriorityOfBest();
			int element = pq.next();
			sb.append("" + element);
			sb.append(" : ");
			sb.append(f.format(priority));
			if (numKeysPrinted < size() - 1) sb.append(multiline ? "\n" : ", ");
			numKeysPrinted++;
		}
		if (numKeysPrinted < size()) sb.append("...");
		if (!multiline) sb.append("]");
		return sb.toString();
	}

	/**
	 * Returns a counter whose keys are the elements in this priority queue, and
	 * whose counts are the priorities in this queue. In the event there are
	 * multiple instances of the same element in the queue, the counter's count
	 * will be the sum of the instances' priorities.
	 * 
	 * @return
	 */
	public Counter<Integer> asCounter() {
		IntPriorityQueue pq = clone();
		Counter<Integer> counter = new Counter<Integer>();
		while (pq.hasNext()) {
			double priority = pq.getPriorityOfBest();
			int element = pq.next();
			counter.incrementCount(element, priority);
		}
		return counter;
	}

	public int[] toSortedList() {
		int[] l = new int[size()];
		IntPriorityQueue pq = clone();
		int k = 0;
		while (pq.hasNext()) {
			l[k++] = pq.next();
		}
		return l;
	}

	/**
	 * Returns the priority of a key if it is already in the queue, or
	 * Double.NaN if it is not.
	 * 
	 * @param element
	 * @return
	 */
	public double getPriorityOfElement(int key) {
		int loc = elementPositions[key];
		if (loc < 0 || loc >= size) return Double.NaN;
		return priorities[loc];
	}

	/**
	 * Promotes a key in the heap.
	 * 
	 * @param element
	 * @param cost
	 */
	public void increaseKey(int element, double cost) {
		int loc = elementPositions[element];
		assert loc >= 0;
		assert cost < priorities[loc];
		priorities[loc] = cost;
		heapifyDown(loc);
	}

	/**
	 * Returns a clone of this priority queue. Modifications to one will not
	 * affect modifications to the other.
	 */
	@Override
	public IntPriorityQueue clone() {
		IntPriorityQueue clonePQ = new IntPriorityQueue(maxElem);
		clonePQ.size = size;
		clonePQ.capacity = capacity;
		clonePQ.elements = CollectionUtils.copyOf(elements, elements.length);
		clonePQ.priorities = CollectionUtils.copyOf(priorities, priorities.length);
		System.arraycopy(elementPositions, 0, clonePQ.elementPositions, 0, elementPositions.length);
		return clonePQ;
	}

	private void grow(int newCapacity) {
		elements = elements == null ? new int[newCapacity] : CollectionUtils.copyOf(elements, newCapacity);
		priorities = priorities == null ? new double[newCapacity] : CollectionUtils.copyOf(priorities, newCapacity);

		capacity = newCapacity;
	}

	private int parent(int loc) {
		return (loc - 1) / 2;
	}

	private int leftChild(int loc) {
		return 2 * loc + 1;
	}

	private int rightChild(int loc) {
		return 2 * loc + 2;
	}

	private void heapifyUp(int loc) {
		if (loc == 0) return;
		int parent = parent(loc);
		if (priorities[loc] > priorities[parent]) {
			swap(loc, parent);
			heapifyUp(parent);
		}
	}

	private void heapifyDown(int loc) {
		int max = loc;
		int leftChild = leftChild(loc);
		if (leftChild < size()) {
			double priority = priorities[loc];
			double leftChildPriority = priorities[leftChild];
			if (leftChildPriority > priority) max = leftChild;
			int rightChild = rightChild(loc);
			if (rightChild < size()) {
				double rightChildPriority = priorities[rightChild(loc)];
				if (rightChildPriority > priority && rightChildPriority > leftChildPriority) max = rightChild;
			}
		}
		if (max == loc) return;
		swap(loc, max);
		heapifyDown(max);
	}

	private void swap(int loc1, int loc2) {
		double tempPriority = priorities[loc1];
		int tempElement = (elements[loc1]);
		priorities[loc1] = priorities[loc2];
		final int element2 = elements[loc2];
		elements[loc1] = element2;
		elementPositions[element2] = loc1;
		priorities[loc2] = tempPriority;
		elements[loc2] = (tempElement);
		elementPositions[tempElement] = loc2;
	}

	private void removeFirst() {
		if (size < 1) return;
		swap(0, size - 1);
		elementPositions[elements[size]] = -1;
		size--;
		heapifyDown(0);
	}

	public static void main(String[] args) {
		IntPriorityQueue pq = new IntPriorityQueue(3);
		System.out.println(pq);
		pq.put(1, 1);
		System.out.println(pq);
		pq.put(3, 3);
		System.out.println(pq);
		pq.put(1, 1.1);
		System.out.println(pq);
		pq.put(2, 2);
		System.out.println(pq);
		System.out.println(pq.toString(2, false));
		while (pq.hasNext()) {
			System.out.println(pq.next());
		}
	}

}
