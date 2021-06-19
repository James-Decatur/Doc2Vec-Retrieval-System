Guide/Instructions to run the project.py program.

Training:
For default training (newsgroup) submit the following argument 'python project.py --mode train'
For reuters training submit the following argument 'python project.py --mode train --training_data reuters'
For custom training submit the following argument 'python project.py --mode train --training_data FOLDER_NAME'

Run:
To run the program using a document input submit the following argument 'python project.py --mode run --data FOLDER_NAME --document SOME_TEXT.txt
or
To run the program using a query input submit the following argument 'python project.py --mode run --data FOLDER_NAME --query 'hello world ...'

Eval:
To evaluate  on the default newsgroup dataset submit the following argument 'python project.py --mode eval'
To evaluate  on the reuters dataset submit the following argument 'python project.py --mode eval --test_data reuters'
