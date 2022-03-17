/*
	Input: adversarial example B-1, original malware B2, insertion
	points Lins, nops list Lnops, distance function D(·, ·)
	for i ∈ range(-1, len(B1)) do
		for j ∈ range(i, len(B0)) do
			Calculate O[i][j] using Equation 1
		end for
	end for

To save time consume:
-	try to create a backtrack function that calculates the correct answer
-	Try to avoid the redundant arguments, minimize the range of possible values of function arguments
-	try to optimize the time complexity of one function call

*/



package main

import "fmt"

func main() {
	fmt.Println("test")
}

// backtrack
int P[n];

	profits(int year, int le, int ri) {
		if le>ri:
			return 0
		else:
			
			return max(
				profit(year, le+1,ri) + year*P[le],
				profit(year, le, ri-1) + year*P[ri];
			)
	}

// create a global variable to replace a redundent value N
int N;
	int P[N]:

	profit(int le, int ri) {
		if le>ri:
			return 0
		else:
			int year = N -(ri -le +1) +1;
			return max(
				profit(le+1, ri) + year*P[le],
				profit(le, ri-1) + year*P[ri];
			)
	}

// cache the computed value
int N;
	int P[N];
	int cache[N][N];

	profit(int le, int ri) {
		if le>ri:
			return 0
		if cache[le][ri] != -1:
			return cache[le][ri]
		int year = N - (ri -le +1) +1
		else:
			return cache[le][ri] = max(
				profit(le+1, ri) + year*P[le];
				profit(le, ri +1) + year*P[ri];
			)
	}