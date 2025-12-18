#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
   DO NOT MODIFY BELOW THIS LINE */

#define N 1000
#define MAX_PEAKS 50
#define PEAK_THRESHOLD 0.6f        // Threshold to consider a sample as a peak //

float rand_float(float min, float max) {
    return min + (float)rand() / RAND_MAX * (max - min);
}

// TASK: COMPLETE THIS FUNCTION
void detect_peak_stream(float sample, int index, int detected_peaks[]) {

    // TO DO
    // WRITE LOGIC TO DETECT PEAKS

   float prev, prev_prev; // Consider two extra variables to store previous sample and sample before that // 

    if(index == 0) {   
        prev_prev = 0.0f;   // An initialization for checking peak in first sample  //
        prev = 0.0f;
   }

   if(index == 1) {   
        if(prev > PEAK_THRESHOLD && prev > prev_prev)   // Condition for checking if first sample is a peak //
        detected_peaks[0] = 1; 
   }

   if(index >= 2){
    if(prev > PEAK_THRESHOLD && prev > sample && prev > prev_prev) // Logic to check if previous sample is a peak + Satisying Peak Threshold //
        detected_peaks[index-1] = 1;     
     else 
        detected_peaks[index-1] = 0;  
    }

    prev_prev = prev;
    prev = sample;      
}

// DO NOT MODIFY BELOW THIS LINE

int main() {
    float eeg[N];
    int true_peaks[N] = {0};
    int detected_peaks[N] = {0};
    srand(time(NULL));

    printf("Simulating EEG samples...\n");

    for (int i = 0; i < N; i++) {

        /* Generate baseline EEG */
        eeg[i] = rand_float(0.1f, 0.2f);

        /* Randomly insert peaks */
        if (rand() % 100 < 4) {   // ~4% chance
            eeg[i] = rand_float(0.7f, 1.0f);
            true_peaks[i] = 1;
        }
        detect_peak_stream(eeg[i], i, detected_peaks);
    }

    // EVALUATION
    int true_positive = 0;
    int false_positive = 0;
    int false_negative = 0;
    int total_peaks = 0;

    for (int i = 0; i < N; i++) {
        if (true_peaks[i])
            total_peaks++;
        if (true_peaks[i] && detected_peaks[i])
            true_positive++;
        else if (!true_peaks[i] && detected_peaks[i])
            false_positive++;
        else if (true_peaks[i] && !detected_peaks[i])
            false_negative++;
    }

    printf("Total samples     : %d\n", N);
    printf("Actual peaks      : %d\n", total_peaks);
    printf("True positives    : %d\n", true_positive);
    printf("False positives   : %d\n", false_positive);
    printf("False negatives   : %d\n", false_negative);

    return 0;
}