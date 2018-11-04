# MLModels
Homemade implementations of many common ML models, including neural nets, logistic / linear regression. 

Models all have a similar structure. Most have a weights attribute, which contains information on their parameters.
All models must have a calculate method, which enables the model, with a certain weight configuration, to output a value upon
an input (codifies hypothesis set). All models must also have an err method, which gets the pointwise error b
etween an input and correct output. (Error function). Models also have a learn method which implements the learning algorithm.

All learn methods return it, E_ins
all models have same calling syntax,

