/*
 -----------------------------------------------------------------
 ** Top contributors:
 **   Shiqi Wang and Suman Jana
 ** This file is part of the ReluVal project.
 ** Copyright (c) 2018-2019 by the authors listed in the file LICENSE
 ** and their institutional affiliations.
 ** All rights reserved.
 -----------------------------------------------------------------
 */

#include <stdlib.h>
#include <time.h>

#include "split.h"

#define AVG_WINDOW 5
#define MAX_THREAD 56
#define MIN_DEPTH_PER_THREAD 5 

float rateCert = 0, rateFals = 0, rateAdv = 0, rateUncer = 1;

struct timeval start, curr, finish, last_finish;

/*
 * Check the existance of concrete adversarial examples
 * It takes in the network and input ranges.
 * If a concrete adversarial example is found,
 * return adv_found 1
 */
// int check_adv(struct NNet* nnet, struct Subproblem *subp)
// {

//     float a0[nnet->inputSize];
//     float a1[nnet->inputSize];
//     struct Matrix adv0 = {a0, 1, nnet->inputSize};
//     struct Matrix adv1 = {a1, 1, nnet->inputSize};

//     // trying 10 different samples
//     for (int n=0; n<10; n++){

//         // concrete data point is gender + some point for all other features
//         for (int i=0; i<nnet->inputSize; i++) {
//             if (i == nnet->sens_feature_idx){
//                 a0[i] = nnet->mins[i];  //for PA=0
//                 a1[i] = nnet->maxes[i]; //for PA=1
//             }
//             else {
//                 int upper = (int) subp->input.upper_matrix.data[i];
//                 int lower = (int) subp->input.lower_matrix.data[i];
//                 int middle = n*(lower+upper)/10; // floor

//                 a0[i] = (float) middle;
//                 a1[i] = (float) middle;
//             }
//         }

//         float out0[nnet->outputSize];
//         float out1[nnet->outputSize];
//         struct Matrix output0 = {out0, nnet->outputSize, 1};
//         struct Matrix output1 = {out1, nnet->outputSize, 1};

//         forward_prop(nnet, &adv0, &output0);
//         forward_prop(nnet, &adv1, &output1);

//         // for sigmoid, one output node
//         int out0Pos = (output0.data[0] > 0);
//         int out1Pos = (output1.data[0] > 0);

//         // at any point, this sample is adv, then we can return here
//         // and no need to check further
//         if (out0Pos != out1Pos) {
//             return 1;   //is_adv!
//         } 
//     }

//     // if we made it here, that means no adv sample was found
//     return 0;
// }

// /////////////////////////////////////////////////////////////////////////////////////////////////// ADULT CENSUS 

static const char* workclass_map[] = {
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked"
};

static const char* education_map[] = {
    "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
    "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
    "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"
};

static const char* marital_status_map[] = {
    "Married-civ-spouse", "Divorced", "Never-married", "Separated",
    "Widowed", "Married-spouse-absent", "Married-AF-spouse"
};

static const char* occupation_map[] = {
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-Forces"
};

static const char* relationship_map[] = {
    "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
};

static const char* sex_map[] = { "Female", "Male" };

static const char* race_map[] = {
    "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
};

static const char* native_country_map[] = {
    "United-States", "Cambodia", "England", "Puerto-Rico", "Canada",
    "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece",
    "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy",
    "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland",
    "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti",
    "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand",
    "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong",
    "Holand-Netherlands"
};

const char* decode_bin(float value, float min_val, float max_val, int n_bins) {
    static char buffer[32];
    int idx = (int)round(value);
    float bin_width = (max_val - min_val) / n_bins;
    float midpoint = min_val + (idx + 0.5f) * bin_width;
    snprintf(buffer, sizeof(buffer), "%d", (int)midpoint);
    return buffer;
}

const char* decode_feature(int feature_index, float value) {
    int idx = (int)round(value);
    switch (feature_index) {
        case 1: return workclass_map[idx];
        case 2: return education_map[idx];
        case 4: return marital_status_map[idx];
        case 5: return occupation_map[idx];
        case 6: return relationship_map[idx];
        case 7: return race_map[idx];
        case 8: return sex_map[idx];
        case 9: return decode_bin(value, 0.0f, 100000.0f, 20);  // capital-gain (real max ≈ 99999)
        case 10: return decode_bin(value, 0.0f, 4356.0f, 20);   // capital-loss (real max ≈ 4356)
        case 12: return native_country_map[idx];
        default: {
            static char buffer[32];
            snprintf(buffer, sizeof(buffer), "%.0f", value);
            return buffer;
        }
    }
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}


// Main check_adv function
int check_adv(struct NNet* nnet, struct Subproblem *subp) {
    static int counterexample_count = 0;
    static FILE* ce_file = NULL;

    static const char* feature_names[] = {
        "age",
        "workclass",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country"
    };

    if (ce_file == NULL) {
        ce_file = fopen("FairQuant-Artifact/FairQuant/counterexamples.csv", "w");
        if (!ce_file) {
            printf("Failed to open counterexamples.csv\n");
            return 0;
        }
        for (int i = 0; i < nnet->inputSize; i++) {
            fprintf(ce_file, "%s,", feature_names[i]);
        }
        fprintf(ce_file, "output,decision\n");
        fflush(ce_file);
    }

    float a0[nnet->inputSize];
    float a1[nnet->inputSize];
    struct Matrix adv0 = {a0, 1, nnet->inputSize};
    struct Matrix adv1 = {a1, 1, nnet->inputSize};

    int counterexample = 0;

    for (int n = 0; n < 10; n++) {
        for (int i = 0; i < nnet->inputSize; i++) {
            if (i == nnet->sens_feature_idx) {
                a0[i] = nnet->mins[i];  // PA = 0
                a1[i] = nnet->maxes[i]; // PA = 1
            } else {
                float lower = subp->input.lower_matrix.data[i];
                float upper = subp->input.upper_matrix.data[i];
                float middle = lower + ((float)n / 10.0f) * (upper - lower);
                a0[i] = middle;
                a1[i] = middle;
            }
        }

        float out0[nnet->outputSize];
        float out1[nnet->outputSize];
        struct Matrix output0 = {out0, nnet->outputSize, 1};
        struct Matrix output1 = {out1, nnet->outputSize, 1};

        forward_prop(nnet, &adv0, &output0);
        forward_prop(nnet, &adv1, &output1);

        // Apply sigmoid to get probabilities between 0 and 1
        float sigmoid_out0 = sigmoid(output0.data[0]);
        float sigmoid_out1 = sigmoid(output1.data[0]);
        
        // Decision based on sigmoid output (threshold = 0.5)
        int out0Pos = sigmoid_out0 > 0.5f;
        int out1Pos = sigmoid_out1 > 0.5f;

        // int out0Pos = output0.data[0] > 0;
        // int out1Pos = output1.data[0] > 0;

        if (out0Pos != out1Pos) {
            counterexample_count++;

            // PA = 0
            for (int i = 0; i < nnet->inputSize; i++) {
                fprintf(ce_file, "%s,", decode_feature(i, a0[i]));
            }
            fprintf(ce_file, "%.6f,%s\n", sigmoid_out0, out0Pos ? "POSITIVE" : "NEGATIVE");

            // PA = 1
            for (int i = 0; i < nnet->inputSize; i++) {
                fprintf(ce_file, "%s,", decode_feature(i, a1[i]));
            }
            fprintf(ce_file, "%.6f,%s\n", sigmoid_out1, out1Pos ? "POSITIVE" : "NEGATIVE");

            fflush(ce_file);
            counterexample++;
            return 1; 
        }
    }

    // if(counterexample == 0){
    //     // fprintf(stdout, "TIDAK ADA\n");
    // }

    // return counterexample; // No counterexample found
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////// GERMAN CREDIT

// static const char* status_map[] = {
//     "A11", "A12", "A13", "A14"  // checking account status
// };

// static const char* credit_history_map[] = {
//     "A30", "A31", "A32", "A33", "A34"  // credit history
// };

// static const char* purpose_map[] = {
//     "A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410"  // purpose
// };

// static const char* savings_map[] = {
//     "A61", "A62", "A63", "A64", "A65"  // savings account/bonds
// };

// static const char* employment_map[] = {
//     "A71", "A72", "A73", "A74", "A75"  // present employment since
// };

// static const char* other_debtors_map[] = {
//     "A101", "A102", "A103"  // other debtors / guarantors
// };

// static const char* property_map[] = {
//     "A121", "A122", "A123", "A124"  // property
// };

// static const char* installment_plans_map[] = {
//     "A141", "A142", "A143"  // other installment plans
// };

// static const char* housing_map[] = {
//     "A151", "A152", "A153"  // housing
// };

// static const char* skill_level_map[] = {
//     "A171", "A172", "A173", "A174"  // job/skill level
// };

// static const char* telephone_map[] = {
//     "A191", "A192"  // telephone
// };

// static const char* foreign_worker_map[] = {
//     "A201", "A202"  // foreign worker
// };

// static const char* sex_map[] = {
//     "A91", "A92", "A93", "A94", "A95"  // personal status and sex
// };

// const char* decode_bin(float value, float min_val, float max_val, int n_bins) {
//     static char buffer[32];
//     int idx = (int)round(value);
//     float bin_width = (max_val - min_val) / n_bins;
//     float midpoint = min_val + (idx + 0.5f) * bin_width;
//     snprintf(buffer, sizeof(buffer), "%d", (int)midpoint);
//     return buffer;
// }

// const char* decode_feature(int feature_index, float value) {
//     int idx = (int)round(value);
//     switch (feature_index) {
//         case 0: return status_map[idx];                                    // status
//         //case 1: return decode_bin(value, 1.0f, 72.0f, 20);               // month (duration)
//         case 2: return credit_history_map[idx];                           // credit_history
//         case 3: return purpose_map[idx];                                  // purpose
//         //case 4: return decode_bin(value, 250.0f, 18424.0f, 20);         // credit_amount
//         case 5: return savings_map[idx];                                  // savings
//         case 6: return employment_map[idx];                               // employment
//         //case 7: return decode_bin(value, 1.0f, 4.0f, 4);                // investment_as_income_percentage
//         case 8: return other_debtors_map[idx];                           // other_debtors
//         //case 9: return decode_bin(value, 1.0f, 4.0f, 4);                // residence_since
//         case 10: return property_map[idx];                               // property
//         //case 11: return decode_bin(value, 19.0f, 75.0f, 20);            // age
//         case 12: return installment_plans_map[idx];                      // installment_plans
//         case 13: return housing_map[idx];                                // housing
//         //case 14: return decode_bin(value, 1.0f, 4.0f, 4);               // number_of_credits
//         case 15: return skill_level_map[idx];                            // skill_level
//         //case 16: return decode_bin(value, 1.0f, 2.0f, 2);               // people_liable_for
//         case 17: return telephone_map[idx];                              // telephone
//         case 18: return foreign_worker_map[idx];                         // foreign_worker
//         case 19: return sex_map[idx];                                    // sex (personal status and sex)
//         default: {
//             static char buffer[32];
//             snprintf(buffer, sizeof(buffer), "%.0f", value);
//             return buffer;
//         }
//     }
// }

// float sigmoid(float x) {
//     return 1.0f / (1.0f + expf(-x));
// }

// // Main check_adv function
// int check_adv(struct NNet* nnet, struct Subproblem *subp) {
//     static int counterexample_count = 0;
//     static FILE* ce_file = NULL;
//     static const char* feature_names[] = {
//         "status",
//         "month",
//         "credit_history",
//         "purpose",
//         "credit_amount",
//         "savings",
//         "employment",
//         "investment_as_income_percentage",
//         "other_debtors",
//         "residence_since",
//         "property",
//         "age",
//         "installment_plans",
//         "housing",
//         "number_of_credits",
//         "skill_level",
//         "people_liable_for",
//         "telephone",
//         "foreign_worker",
//         "sex"
//     };
//     if (ce_file == NULL) {
//         ce_file = fopen("FairQuant-Artifact/FairQuant/counterexamples.csv", "w");
//         if (!ce_file) {
//             printf("Failed to open counterexamples.csv\n");
//             return 0;
//         }
//         for (int i = 0; i < nnet->inputSize; i++) {
//             fprintf(ce_file, "%s,", feature_names[i]);
//         }
//         fprintf(ce_file, "output,decision\n");
//         fflush(ce_file);
//     }
//     float a0[nnet->inputSize];
//     float a1[nnet->inputSize];
//     struct Matrix adv0 = {a0, 1, nnet->inputSize};
//     struct Matrix adv1 = {a1, 1, nnet->inputSize};
//     int counterexample = 0;
//     for (int n = 0; n < 10; n++) {
//         for (int i = 0; i < nnet->inputSize; i++) {
//             if (i == nnet->sens_feature_idx) {
//                 a0[i] = nnet->mins[i];  // PA = 0
//                 a1[i] = nnet->maxes[i]; // PA = 1
//             } else {
//                 float lower = subp->input.lower_matrix.data[i];
//                 float upper = subp->input.upper_matrix.data[i];
//                 float middle = lower + ((float)n / 10.0f) * (upper - lower);
//                 a0[i] = middle;
//                 a1[i] = middle;
//             }
//         }
//         float out0[nnet->outputSize];
//         float out1[nnet->outputSize];
//         struct Matrix output0 = {out0, nnet->outputSize, 1};
//         struct Matrix output1 = {out1, nnet->outputSize, 1};
//         forward_prop(nnet, &adv0, &output0);
//         forward_prop(nnet, &adv1, &output1);

//         // Apply sigmoid to get probabilities between 0 and 1
//         float sigmoid_out0 = sigmoid(output0.data[0]);
//         float sigmoid_out1 = sigmoid(output1.data[0]);
        
//         // Decision based on sigmoid output (threshold = 0.5)
//         int out0Pos = sigmoid_out0 > 0.5f;
//         int out1Pos = sigmoid_out1 > 0.5f;

//         // int out0Pos = output0.data[0] > 0;
//         // int out1Pos = output1.data[0] > 0;

//         if (out0Pos != out1Pos) {
//             counterexample_count++;
//             // PA = 0
//             for (int i = 0; i < nnet->inputSize; i++) {
//                 fprintf(ce_file, "%s,", decode_feature(i, a0[i]));
//             }
//             fprintf(ce_file, "%.6f,%s\n", sigmoid_out0, out0Pos ? "POSITIVE" : "NEGATIVE");
//             // PA = 1
//             for (int i = 0; i < nnet->inputSize; i++) {
//                 fprintf(ce_file, "%s,", decode_feature(i, a1[i]));
//             }
//             fprintf(ce_file, "%.6f,%s\n",sigmoid_out1, out1Pos ? "POSITIVE" : "NEGATIVE");
//             fflush(ce_file);
//             counterexample++;
//             return 1;
//         }
//     }
//     // if(counterexample == 0){
//     //     // fprintf(stdout, "TIDAK ADA\n");
//     // }
//     // return counterexample; // No counterexample found
//     return 0;
// }

/////////////////////////////////////////////////////////////////////////////////////////////////// BANK MARKETING

// static const char* job_map[] = {
//     "admin.", "blue-collar", "entrepreneur", "housemaid", "management", 
//     "retired", "self-employed", "services", "student", "technician", 
//     "unemployed", "unknown"
// };

// static const char* marital_map[] = {
//     "divorced", "married", "single", "unknown"
// };

// static const char* education_map[] = {
//     "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", 
//     "professional.course", "university.degree", "unknown"
// };

// static const char* default_map[] = {
//     "no", "unknown", "yes"
// };

// static const char* housing_map[] = {
//     "no", "unknown", "yes"
// };

// static const char* loan_map[] = {
//     "no", "unknown", "yes"
// };

// static const char* contact_map[] = {
//     "cellular", "telephone"
// };

// static const char* month_map[] = {
//     "jan", "feb", "mar", "apr", "may", "jun", 
//     "jul", "aug", "sep", "oct", "nov", "dec"
// };

// static const char* day_of_week_map[] = {
//     "mon", "tue", "wed", "thu", "fri"
// };

// static const char* poutcome_map[] = {
//     "failure", "nonexistent", "success"
// };

// const char* decode_bin(float value, float min_val, float max_val, int n_bins) {
//     static char buffer[32];
//     int idx = (int)round(value);
//     float bin_width = (max_val - min_val) / n_bins;
//     float midpoint = min_val + (idx + 0.5f) * bin_width;
//     snprintf(buffer, sizeof(buffer), "%.1f", midpoint);
//     return buffer;
// }

// const char* decode_feature(int feature_index, float value) {
//     int idx = (int)round(value);
//     switch (feature_index) {
//         case 0: return job_map[idx];                                     // job
//         case 1: return marital_map[idx];                                 // marital
//         case 2: return education_map[idx];                               // education
//         case 3: return default_map[idx];                                 // default
//         case 4: return housing_map[idx];                                 // housing
//         case 5: return loan_map[idx];                                    // loan
//         case 6: return contact_map[idx];                                 // contact
//         case 7: return month_map[idx];                                   // month
//         case 8: return day_of_week_map[idx];                             // day_of_week
//         // case 9: return decode_bin(value, -3.4f, 1.4f, 20);               // emp.var.rate
//         // case 10: return decode_bin(value, 0.0f, 4918.0f, 50);            // duration
//         // case 11: return decode_bin(value, 1.0f, 56.0f, 20);              // campaign
//         // case 12: return decode_bin(value, 0.0f, 999.0f, 20);             // pdays
//         // case 13: return decode_bin(value, 0.0f, 7.0f, 8);                // previous
//         case 14: return poutcome_map[idx];                               // poutcome
//         // case 15: return decode_bin(value, 17.0f, 98.0f, 20);             // age
//         default: {
//             static char buffer[32];
//             snprintf(buffer, sizeof(buffer), "%.0f", value);
//             return buffer;
//         }
//     }
// }

// float sigmoid(float x) {
//     return 1.0f / (1.0f + expf(-x));
// }

// // Main check_adv function
// int check_adv(struct NNet* nnet, struct Subproblem *subp) {
//     static int counterexample_count = 0;
//     static FILE* ce_file = NULL;
//     static const char* feature_names[] = {
//         "job",
//         "marital",
//         "education",
//         "default",
//         "housing",
//         "loan",
//         "contact",
//         "month",
//         "day_of_week",
//         "emp.var.rate",
//         "duration",
//         "campaign",
//         "pdays",
//         "previous",
//         "poutcome",
//         "age",
//     };
//     if (ce_file == NULL) {
//         ce_file = fopen("FairQuant-Artifact/FairQuant/counterexamples.csv", "w");
//         if (!ce_file) {
//             printf("Failed to open counterexamples.csv\n");
//             return 0;
//         }
//         for (int i = 0; i < nnet->inputSize; i++) {
//             fprintf(ce_file, "%s,", feature_names[i]);
//         }
//         fprintf(ce_file, "output,decision\n");
//         fflush(ce_file);
//     }
//     float a0[nnet->inputSize];
//     float a1[nnet->inputSize];
//     struct Matrix adv0 = {a0, 1, nnet->inputSize};
//     struct Matrix adv1 = {a1, 1, nnet->inputSize};
//     int counterexample = 0;
//     for (int n = 0; n < 10; n++) {
//         for (int i = 0; i < nnet->inputSize; i++) {
//             if (i == nnet->sens_feature_idx) {
//                 a0[i] = nnet->mins[i];  // PA = 0
//                 a1[i] = nnet->maxes[i]; // PA = 1
//             } else {
//                 float lower = subp->input.lower_matrix.data[i];
//                 float upper = subp->input.upper_matrix.data[i];
//                 float middle = lower + ((float)n / 10.0f) * (upper - lower);
//                 a0[i] = middle;
//                 a1[i] = middle;
//             }
//         }
//         float out0[nnet->outputSize];
//         float out1[nnet->outputSize];
//         struct Matrix output0 = {out0, nnet->outputSize, 1};
//         struct Matrix output1 = {out1, nnet->outputSize, 1};
//         forward_prop(nnet, &adv0, &output0);
//         forward_prop(nnet, &adv1, &output1);

//         // Apply sigmoid to get probabilities between 0 and 1
//         float sigmoid_out0 = sigmoid(output0.data[0]);
//         float sigmoid_out1 = sigmoid(output1.data[0]);
        
//         // Decision based on sigmoid output (threshold = 0.5)
//         int out0Pos = sigmoid_out0 > 0.5f;
//         int out1Pos = sigmoid_out1 > 0.5f;

//         // int out0Pos = output0.data[0] > 0;
//         // int out1Pos = output1.data[0] > 0;

//         if (out0Pos != out1Pos) {
//             counterexample_count++;
//             // PA = 0
//             for (int i = 0; i < nnet->inputSize; i++) {
//                 fprintf(ce_file, "%s,", decode_feature(i, a0[i]));
//             }
//             fprintf(ce_file, "%.6f,%s\n", sigmoid_out0, out0Pos ? "POSITIVE" : "NEGATIVE");
//             // PA = 1
//             for (int i = 0; i < nnet->inputSize; i++) {
//                 fprintf(ce_file, "%s,", decode_feature(i, a1[i]));
//             }
//             fprintf(ce_file, "%.6f,%s\n", sigmoid_out1, out1Pos ? "POSITIVE" : "NEGATIVE");
//             fflush(ce_file);
//             counterexample++;
//             return 1;
//         }
//     }
//     // if(counterexample == 0){
//     //     // fprintf(stdout, "TIDAK ADA\n");
//     // }
//     // return counterexample; // No counterexample found
//     return 0;
// }

/////////////////////////////////////////////////////////////////////////////////////////////////// DEFAULT_CREDIT (NEW)

// float sigmoid(float x) {
//     return 1.0f / (1.0f + expf(-x));
// }

// // Approximate unbinning from bin index (0–4)
// float unbin_continuous(int feature_index, float bin_value) {
//     int bin = (int)round(bin_value);
//     if (bin < 0) bin = 0;
//     if (bin > 4) bin = 4;

//     switch (feature_index) {
//         case 0: { // LIMIT_BAL
//             float map[5] = {10000, 80000, 150000, 250000, 500000};
//             return map[bin];
//         }
//         case 4: { // AGE
//             float map[5] = {20, 30, 40, 50, 60};
//             return map[bin];
//         }
//         case 11: case 12: case 13: case 14: case 15: case 16: { // BILL_AMT1–6
//             float map[5] = {0, 10000, 30000, 70000, 200000};
//             return map[bin];
//         }
//         case 17: case 18: case 19: case 20: case 21: case 22: { // PAY_AMT1–6
//             float map[5] = {0, 500, 2000, 5000, 20000};
//             return map[bin];
//         }
//         default:
//             return bin_value; // categorical stays same
//     }
// }

// // Decode raw numeric feature (unbinned)
// float decode_feature_numeric(int feature_index, float value) {
//     switch (feature_index) {
//         case 0:  // LIMIT_BAL
//         case 4:  // AGE
//         case 11: case 12: case 13: case 14: case 15: case 16: // BILL_AMT*
//         case 17: case 18: case 19: case 20: case 21: case 22: // PAY_AMT*
//             return unbin_continuous(feature_index, value);
//         default:
//             return value;
//     }
// }

// // Main check_adv function (raw + unbinned output)
// int check_adv(struct NNet* nnet, struct Subproblem *subp) {
//     static int counterexample_count = 0;
//     static FILE* ce_file = NULL;
//     static int initialized = 0;

//     if (!initialized) {
//         srand(time(NULL));
//         initialized = 1;
//     }

//     static const char* feature_names[] = {
//         "LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE",
//         "PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
//         "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
//         "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"
//     };

//     if (ce_file == NULL) {
//         ce_file = fopen("FairQuant-Artifact/FairQuant/counterexamples.csv", "w");
//         if (!ce_file) {
//             printf("Failed to open counterexamples.csv\n");
//             return 0;
//         }
//         for (int i = 0; i < nnet->inputSize; i++) {
//             fprintf(ce_file, "%s,", feature_names[i]);
//         }
//         fprintf(ce_file, "output,decision\n");
//         fflush(ce_file);
//     }

//     float a0[nnet->inputSize];
//     float a1[nnet->inputSize];
//     struct Matrix adv0 = {a0, 1, nnet->inputSize};
//     struct Matrix adv1 = {a1, 1, nnet->inputSize};

//     int counterexample = 0;

//     for (int n = 0; n < 1; n++) {
//         for (int i = 0; i < nnet->inputSize; i++) {
//             if (i == nnet->sens_feature_idx) {
//                 a0[i] = nnet->mins[i];  // Female (SEX=0)
//                 a1[i] = nnet->maxes[i]; // Male (SEX=1)
//             } else {
//                 float lower = subp->input.lower_matrix.data[i];
//                 float upper = subp->input.upper_matrix.data[i];
//                 float middle = lower + ((float)rand() / RAND_MAX) * (upper - lower);
//                 a0[i] = middle;
//                 a1[i] = middle;
//             }
//         }

//         float out0[nnet->outputSize];
//         float out1[nnet->outputSize];
//         struct Matrix output0 = {out0, nnet->outputSize, 1};
//         struct Matrix output1 = {out1, nnet->outputSize, 1};

//         forward_prop(nnet, &adv0, &output0);
//         forward_prop(nnet, &adv1, &output1);

//         float sigmoid_out0 = sigmoid(output0.data[0]);
//         float sigmoid_out1 = sigmoid(output1.data[0]);

//         int out0Pos = sigmoid_out0 > 0.5f;
//         int out1Pos = sigmoid_out1 > 0.5f;

//         if (out0Pos != out1Pos) {
//             counterexample_count++;

//             for (int i = 0; i < nnet->inputSize; i++) {
//                 fprintf(ce_file, "%.0f,", decode_feature_numeric(i, a0[i]));
//             }
//             fprintf(ce_file, "%.6f,%d\n", sigmoid_out0, out0Pos);

//             for (int i = 0; i < nnet->inputSize; i++) {
//                 fprintf(ce_file, "%.0f,", decode_feature_numeric(i, a1[i]));
//             }
//             fprintf(ce_file, "%.6f,%d\n", sigmoid_out1, out1Pos);

//             fflush(ce_file);
//             counterexample++;
//             return 1;
//         }
//     }

//     return counterexample;
// }

/////////////////////////////////////////////////////////////////////////////////////////////////// COMPAS

// float sigmoid(float x) {
//     return 1.0f / (1.0f + expf(-x));
// }

// // COMPAS features are already preprocessed/binned, so no unbinning needed
// float decode_feature_numeric(int feature_index, float value) {
//     // For COMPAS, all features are already in their final form (0/1 or small integers)
//     return value;
// }

// // Main check_adv function for COMPAS dataset
// int check_adv(struct NNet* nnet, struct Subproblem *subp) {
//     static int counterexample_count = 0;
//     static FILE* ce_file = NULL;
//     static int initialized = 0;

//     if (!initialized) {
//         srand(time(NULL));
//         initialized = 1;
//     }

//     // COMPAS feature names (5 features total)
//     static const char* feature_names[] = {
//         "Two_yr_Recidivism",
//         "Number_of_Priors", 
//         "Age",
//         "Race",
//         "Female",
//         "Misdemeanor"
//     };

//     if (ce_file == NULL) {
//         ce_file = fopen("counterexamples.csv", "w");
//         if (!ce_file) {
//             printf("Failed to open counterexamples.csv\n");
//             return 0;
//         }
//         // Write header
//         for (int i = 0; i < nnet->inputSize; i++) {
//             fprintf(ce_file, "%s,", feature_names[i]);
//         }
//         fprintf(ce_file, "output,decision\n");
//         fflush(ce_file);
//     }

//     float a0[nnet->inputSize];
//     float a1[nnet->inputSize];
//     struct Matrix adv0 = {a0, 1, nnet->inputSize};
//     struct Matrix adv1 = {a1, 1, nnet->inputSize};

//     int counterexample = 0;

//     for (int n = 0; n < 1; n++) {
//         for (int i = 0; i < nnet->inputSize; i++) {
//             if (i == nnet->sens_feature_idx) {
//                 // Race is the sensitive feature (index 3)
//                 a0[i] = nnet->mins[i];  // White (Race=0)
//                 a1[i] = nnet->maxes[i]; // Non-White (Race=1)
//             } else {
//                 // Sample randomly from the subproblem bounds
//                 float lower = subp->input.lower_matrix.data[i];
//                 float upper = subp->input.upper_matrix.data[i];
//                 float middle = lower + ((float)rand() / RAND_MAX) * (upper - lower);
//                 a0[i] = middle;
//                 a1[i] = middle;
//             }
//         }

//         float out0[nnet->outputSize];
//         float out1[nnet->outputSize];
//         struct Matrix output0 = {out0, nnet->outputSize, 1};
//         struct Matrix output1 = {out1, nnet->outputSize, 1};

//         forward_prop(nnet, &adv0, &output0);
//         forward_prop(nnet, &adv1, &output1);

//         float sigmoid_out0 = sigmoid(output0.data[0]);
//         float sigmoid_out1 = sigmoid(output1.data[0]);

//         int out0Pos = sigmoid_out0 > 0.5f;
//         int out1Pos = sigmoid_out1 > 0.5f;

//         // Found a counterexample (different predictions for White vs Non-White)
//         if (out0Pos != out1Pos) {
//             counterexample_count++;

//             // Write White individual
//             for (int i = 0; i < nnet->inputSize; i++) {
//                 fprintf(ce_file, "%.0f,", decode_feature_numeric(i, a0[i]));
//             }
//             fprintf(ce_file, "%.6f,%d\n", sigmoid_out0, out0Pos);

//             // Write Non-White individual
//             for (int i = 0; i < nnet->inputSize; i++) {
//                 fprintf(ce_file, "%.0f,", decode_feature_numeric(i, a1[i]));
//             }
//             fprintf(ce_file, "%.6f,%d\n", sigmoid_out1, out1Pos);

//             fflush(ce_file);
//             counterexample++;
//             return 1;
//         }
//     }

//     return counterexample;
// }

/////////////////////////////////////////////////////////////////////////////////////////////////// LSAC

// float sigmoid(float x) {
//     return 1.0f / (1.0f + expf(-x));
// }

// // LSAC features are already preprocessed/binned, so no unbinning needed
// float decode_feature_numeric(int feature_index, float value) {
//     // For LSAC, all features are already in their final form (binned 0-4 or binary 0-1)
//     return value;
// }

// // Main check_adv function for LSAC dataset
// int check_adv(struct NNet* nnet, struct Subproblem *subp) {
//     static int counterexample_count = 0;
//     static FILE* ce_file = NULL;
//     static int initialized = 0;

//     if (!initialized) {
//         srand(time(NULL));
//         initialized = 1;
//     }

//     // LSAC feature names (5 features total)
//     static const char* feature_names[] = {
//         "lsat",
//         "ugpa",
//         "race",
//         "male",
//         "fam_inc"
//     };

//     if (ce_file == NULL) {
//         ce_file = fopen("FairQuant-Artifact/FairQuant/counterexamples.csv", "w");
//         if (!ce_file) {
//             printf("Failed to open counterexamples.csv\n");
//             return 0;
//         }
//         // Write header
//         for (int i = 0; i < nnet->inputSize; i++) {
//             fprintf(ce_file, "%s,", feature_names[i]);
//         }
//         fprintf(ce_file, "output,decision\n");
//         fflush(ce_file);
//     }

//     float a0[nnet->inputSize];
//     float a1[nnet->inputSize];
//     struct Matrix adv0 = {a0, 1, nnet->inputSize};
//     struct Matrix adv1 = {a1, 1, nnet->inputSize};

//     int counterexample = 0;

//     for (int n = 0; n < 1; n++) {
//         for (int i = 0; i < nnet->inputSize; i++) {
//             if (i == nnet->sens_feature_idx) {
//                 // Race is the sensitive feature (index 2)
//                 a0[i] = nnet->mins[i];  // White (race=0)
//                 a1[i] = nnet->maxes[i]; // Non-White (race=1)
//             } else {
//                 // Sample randomly from the subproblem bounds
//                 float lower = subp->input.lower_matrix.data[i];
//                 float upper = subp->input.upper_matrix.data[i];
//                 float middle = lower + ((float)rand() / RAND_MAX) * (upper - lower);
//                 a0[i] = middle;
//                 a1[i] = middle;
//             }
//         }

//         float out0[nnet->outputSize];
//         float out1[nnet->outputSize];
//         struct Matrix output0 = {out0, nnet->outputSize, 1};
//         struct Matrix output1 = {out1, nnet->outputSize, 1};

//         forward_prop(nnet, &adv0, &output0);
//         forward_prop(nnet, &adv1, &output1);

//         float sigmoid_out0 = sigmoid(output0.data[0]);
//         float sigmoid_out1 = sigmoid(output1.data[0]);

//         int out0Pos = sigmoid_out0 > 0.5f;
//         int out1Pos = sigmoid_out1 > 0.5f;

//         // Found a counterexample (different predictions for White vs Non-White)
//         if (out0Pos != out1Pos) {
//             counterexample_count++;

//             // Write White individual
//             for (int i = 0; i < nnet->inputSize; i++) {
//                 fprintf(ce_file, "%.0f,", decode_feature_numeric(i, a0[i]));
//             }
//             fprintf(ce_file, "%.6f,%d\n", sigmoid_out0, out0Pos);

//             // Write Non-White individual
//             for (int i = 0; i < nnet->inputSize; i++) {
//                 fprintf(ce_file, "%.0f,", decode_feature_numeric(i, a1[i]));
//             }
//             fprintf(ce_file, "%.6f,%d\n", sigmoid_out1, out1Pos);

//             fflush(ce_file);
//             counterexample++;
//             return 1;
//         }
//     }

//     return counterexample;
// }

/////////////////////////////////////////////////////////////////////////////////////////////////// UCI 

// float sigmoid(float x) {
//     return 1.0f / (1.0f + expf(-x));
// }

// // Approximate unbinning from bin index
// float unbin_continuous(int feature_index, float bin_value) {
//     int bin = (int)round(bin_value);
    
//     switch (feature_index) {
//         case 2: { // age (binned to 0-1)
//             if (bin < 0) bin = 0;
//             if (bin > 1) bin = 1;
//             float map[2] = {17, 20}; // Approximate age ranges
//             return map[bin];
//         }
//         case 29: { // absences (binned to 0-4)
//             if (bin < 0) bin = 0;
//             if (bin > 4) bin = 4;
//             float map[5] = {0, 2, 4, 8, 20}; // Approximate absence ranges
//             return map[bin];
//         }
//         default:
//             return bin_value; // categorical/ordinal stays same
//     }
// }

// // Decode raw numeric feature (unbinned)
// float decode_feature_numeric(int feature_index, float value) {
//     switch (feature_index) {
//         case 2:  // age
//         case 29: // absences
//             return unbin_continuous(feature_index, value);
//         default:
//             return value;
//     }
// }

// // Main check_adv function (raw + unbinned output)
// int check_adv(struct NNet* nnet, struct Subproblem *subp) {
//     static int counterexample_count = 0;
//     static FILE* ce_file = NULL;
//     static int initialized = 0;

//     if (!initialized) {
//         srand(time(NULL));
//         initialized = 1;
//     }

//     static const char* feature_names[] = {
//         "school","sex","age","address","famsize","Pstatus",
//         "Medu","Fedu","Mjob","Fjob","reason","guardian",
//         "traveltime","studytime","failures","schoolsup","famsup",
//         "paid","activities","nursery","higher","internet","romantic",
//         "famrel","freetime","goout","Dalc","Walc","health","absences"
//     };

//     if (ce_file == NULL) {
//         ce_file = fopen("FairQuant-Artifact/FairQuant/counterexamples.csv", "w");
//         if (!ce_file) {
//             printf("Failed to open counterexamples.csv\n");
//             return 0;
//         }
//         for (int i = 0; i < nnet->inputSize; i++) {
//             fprintf(ce_file, "%s,", feature_names[i]);
//         }
//         fprintf(ce_file, "output,decision\n");
//         fflush(ce_file);
//     }

//     float a0[nnet->inputSize];
//     float a1[nnet->inputSize];
//     struct Matrix adv0 = {a0, 1, nnet->inputSize};
//     struct Matrix adv1 = {a1, 1, nnet->inputSize};

//     int counterexample = 0;

//     for (int n = 0; n < 1; n++) {
//         for (int i = 0; i < nnet->inputSize; i++) {
//             if (i == nnet->sens_feature_idx) {
//                 a0[i] = nnet->mins[i];  // Female (sex=0)
//                 a1[i] = nnet->maxes[i]; // Male (sex=1)
//             } else {
//                 float lower = subp->input.lower_matrix.data[i];
//                 float upper = subp->input.upper_matrix.data[i];
//                 float middle = lower + ((float)rand() / RAND_MAX) * (upper - lower);
//                 a0[i] = middle;
//                 a1[i] = middle;
//             }
//         }

//         float out0[nnet->outputSize];
//         float out1[nnet->outputSize];
//         struct Matrix output0 = {out0, nnet->outputSize, 1};
//         struct Matrix output1 = {out1, nnet->outputSize, 1};

//         forward_prop(nnet, &adv0, &output0);
//         forward_prop(nnet, &adv1, &output1);

//         float sigmoid_out0 = sigmoid(output0.data[0]);
//         float sigmoid_out1 = sigmoid(output1.data[0]);

//         int out0Pos = sigmoid_out0 > 0.5f;
//         int out1Pos = sigmoid_out1 > 0.5f;

//         if (out0Pos != out1Pos) {
//             counterexample_count++;

//             for (int i = 0; i < nnet->inputSize; i++) {
//                 fprintf(ce_file, "%.0f,", decode_feature_numeric(i, a0[i]));
//             }
//             fprintf(ce_file, "%.6f,%d\n", sigmoid_out0, out0Pos);

//             for (int i = 0; i < nnet->inputSize; i++) {
//                 fprintf(ce_file, "%.0f,", decode_feature_numeric(i, a1[i]));
//             }
//             fprintf(ce_file, "%.6f,%d\n", sigmoid_out1, out1Pos);

//             fflush(ce_file);
//             counterexample++;
//             return 1;
//         }
//     }

//     return counterexample;
// }