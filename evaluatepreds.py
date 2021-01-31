def evaluate_preds(model, X_train, X_test, y_train, y_test):
    """Валидация модели, вывод отчетов"""
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    """ Фикс бага"""
    """Если вылезает хуй пойми какая ошибка, раскоментировать код ниже, а колонку 'churn' заменить на ту, которую предсказываешь"""
    
#     list = [] 
#     for el in y_test['churn']:
#         list.append(el)
#     list = np.array(list)
#     y_test = list
    
    
    cv_score = cross_val_score(
        model,
        X_train,
        y_train,
        scoring='f1',
        cv=StratifiedKFold(
            n_splits=3,
            random_state=42,
            shuffle=True
        )
    )
    
    get_classification_report(y_train, y_train_pred, y_test, y_test_pred, cv_score)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred, pos_label=1)
    plt.rcParams['figure.figsize'] = 5, 5
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], color='grey', linestyle='dashed')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    
    disp = plot_precision_recall_curve(model, X_test, y_test)
    disp.ax_.set_title('Precision-Recall curve')
    
def get_classification_report(y_train_true, y_train_pred, y_test_true, y_test_pred, cv_score):
    """Отчет с метриками модели"""
    
    print('Train\n\n' + classification_report(y_train_true, y_train_pred))
    print('Test\n\n' + classification_report(y_test_true, y_test_pred))
    print('Confusion Matrix\n')
    print(pd.crosstab(y_test_true, y_test_pred))
    print('\nCross Validation Score: ' + str(round(cv_score.mean(),3)))
