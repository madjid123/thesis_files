<!-- \chResults and Discussion} \label{chap-4} -->
# Results

This section presents the results of both the multi-label classification part and the ligand-based process. The results are based on evaluation metrics previously presented in \cref{chap-3}

## Classification Results

\begin{table}[H]
\begin{center}
\resizebox{\linewidth}{!}{
\begin{tabular}{ccc|ccc|ccc}
\multicolumn{3}{c}{DS1}&\multicolumn{3}{c}{DS2}&\multicolumn{3}{c}{DS3}\\
\hline
\rowcolor[HTML]{FFE5B6}
\textbf{Activity Index} & {\color[HTML]{000000} \textbf{Train partition}} & {\color[HTML]{000000} \textbf{Test partition}} & \textbf{Activity Index} & {\color[HTML]{000000} \textbf{Train partition}} & {\color[HTML]{000000} \textbf{Test partition}} & \textbf{Activity Index} & {\color[HTML]{000000} \textbf{Train partition}} & {\color[HTML]{000000} \textbf{Test partition}}\\ \hline
31420          & 901  & 224  & 7707           & 155  & 32   & 9249           & 704  & 150     \\
71523          & 584  & 159  & 7708           & 128  & 27   & 12455          & 1071 & 286     \\
37110          & 633  & 160  & 31420          & 901  & 224  & 12464          & 400  & 99      \\
31432          & 768  & 172  & 42710          & 83   & 24   & 31281          & 81   & 23      \\
42731          & 1008 & 233  & 64100          & 1026 & 248  & 43210          & 728  & 213     \\
6233           & 600  & 148  & 64200          & 119  & 36   & 71522          & 555  & 128     \\
6245           & 287  & 72   & 64220          & 797  & 220  & 75721          & 483  & 114     \\
7701           & 303  & 77   & 64500          & 92   & 24   & 78331          & 511  & 113     \\
6235           & 645  & 175  & 64350          & 310  & 69   & 78348          & 456  & 121     \\
78374          & 329  & 110  & 75755          & 301  & 73   & 78351          & 1639 & 394     \\
78331          & 510  & 114  &                &      &      &                &      &         \\ \hline
\textbf{Total} & 6568 & 1644 & \textbf{Total} & 3912 & 977  & \textbf{Total} & 6628 & 1641    \\ \hline
\end{tabular}
}
\caption{Data distribution of classes between train and test}
\label{tab:chapter4.1}
\end{center}
\end{table}

## Training results
This subsection presents the training results of the Proximal Policy Optimization (PPO) agent that was trained to perform multi-label classification of biological activities. The objective is to assess the performance of the agent and evaluate its effectiveness in accurately predicting the presence of various biological activities.

During the training process, the agent's reward demonstrated an overall increasing trend, albeit with occasional periods of noisy decrease. After approximately 7 million time steps, the reward reached a peak value of 31 (see \Cref{subfig:reward}), representing the number of correctly predicted labels. The intermittent decrease in reward can be attributed to the inherently stochastic nature of the training process, where the agent explores different strategies to optimize its policy while encountering fluctuations in its performance.

To further analyze the performance of the PPO agent, we examined the value loss, which quantifies the disparity between the predicted values and the actual values. As illustrated in \Cref{subfig:value_loss}, the value loss exhibited a consistent decrease throughout the training duration. Starting at approximately 50, the value loss gradually declined and stabilized around 10 towards the end of the time steps. This reduction in value loss indicates an improvement in the agent's capability to estimate the actual values associated with specific state-action pairs.


Additionally, we examined the training loss, which captures the overall loss incurred during the training process. \Cref{subfig:loss} displays the training loss, showcasing a notable decrease from around 25 to approximately 5. This reduction in training loss indicates the agent's improved learning and adaptation as it progresses through the training iterations.


Furthermore, we investigated the policy gradient loss, which reflects the discrepancy between the estimated and actual gradients of the agent's policy. The policy gradient loss was characterized by fluctuations and exhibited a noisy pattern throughout the training process. However, the loss remained mostly stable, hovering around a value of 0 (see \Cref{subfig:policy_loss}). This stability suggests that the agent's policy was robust and exhibited minimal volatility during the training iterations.


In summary, the PPO agent's training results for multi-label classification of biological activities demonstrate promising outcomes. The reward steadily increased, with intermittent periods of noisy decrease, ultimately reaching a peak value of 31 at 7 million time steps, denoting the number of accurately predicted labels. The decreasing value loss and training loss reflect the agent's enhanced capability in estimating true values and learning the underlying patterns within the data. Despite the noisy nature of the policy gradient loss, it remained mostly stable around 0, indicating a consistent and reliable policy during the training process.

\begin{figure}[H]
  \centering
  \begin{subfigure}[b]{0.46\textwidth}
    \includegraphics[width=\textwidth]{Images/chap4/Reward.pdf}
    \caption{Average episode reward during training}
    \label{subfig:reward}
  \end{subfigure}
  \begin{subfigure}[b]{0.46\textwidth}
    \includegraphics[width=\textwidth]{Images/chap4/Loss.pdf}
    \caption{Training loss over time}
    \label{subfig:loss}
  \end{subfigure}
  \begin{subfigure}[b]{0.45\textwidth}
    \includegraphics[width=\textwidth]{Images/chap4/Policy Gradient Loss.pdf}
    \caption{Policy gradient loss over time}
    \label{subfig:policy_loss}
  \end{subfigure}
  \begin{subfigure}[b]{0.45\textwidth}
    \includegraphics[width=\textwidth]{Images/chap4/Value Loss.pdf}
    \caption{Value loss over time}
    \label{subfig:value_loss}
  \end{subfigure}
   
  \caption{Performance of an agent trained for 7 million timesteps. The top left plot shows the average episode reward during training, while the top right plot shows the training loss over time. The bottom left plot shows the policy gradient loss over time, and the bottom right plot shows the value loss over time.}
    \label{fig:agent_performance}
\end{figure}


## Test results


To evaluate the performance of the PPO agent trained for multi-label classification of biological activities, we conducted a classification test on the 20\% test portion of the MDDR dataset. The evaluation of the method’s performance on each class and on the entire dataset employs precision, recall, f1-score, and Jaccard-score. Table \ref{tab:class_report} shows the evaluation metrics for each activity class in DS1, DS2, and DS3 of the MDDR dataset.


For DS1, the precision scores for all labels ranged from 0.234 to 0.996, with an average precision of 0.644. The recall scores ranged from 0.958 to 1.000, with an average recall of 0.976. The F1-score ranged from 0.376 to 0.966, with an average F1-score of 0.736. The Jaccard score ranged from 0.232 to 0.987, with an average Jaccard score of 0.667.

Overall, the PPO agent performed well in the classification test, with high precision and recall scores for most labels. However, the agent's performance was weaker for labels 71523, 31432, 6245, 7701, and 78374, as indicated by the lower F1-scores and Jaccard scores for these labels. Further analysis of these labels may be necessary to improve the agent's performance on these tasks.


and for DS2, the precision scores obtained for labels in the DS2 dataset ranged from 0.41 to 1.00, with an average precision of 0.80. The recall scores ranged from 0.93 to 1.00, with an average recall of 0.98. The F1-scores ranged from 0.57 to 0.99, with an average F1-score of 0.83. The Jaccard scores ranged from 0.40 to 0.99, with an average Jaccard score of 0.78.

Overall, the PPO agent demonstrated strong performance in the classification test on the DS2 dataset. Most labels achieved high precision, recall, F1-score, and Jaccard score values, indicating accurate and consistent classification results. Notably, labels 7708, 31420, and 64220 exhibited outstanding performance, with perfect or near-perfect scores across all evaluation metrics.

However, labels 7707, 64100, 64200, 64500, and 64350 presented slightly lower F1-scores and Jaccard scores compared to other labels, suggesting some room for improvement in the agent's classification performance for these specific activity classes.

and for the DS3, the precision scores for the different labels in the DS3 dataset ranged from 0.3426 to 0.9565, with an average precision of 0.6465. The recall scores ranged from 0.9495 to 1.0000, with an average recall of 0.9773. The F1-scores ranged from 0.5051 to 0.9697, with an average F1-score of 0.7666. The Jaccard scores ranged from 0.3379 to 0.9412, with an average Jaccard score of 0.6897.

Overall, the PPO agent demonstrated satisfactory performance in the classification test on the DS3 dataset. Most labels achieved reasonable precision, recall, F1-score, and Jaccard score values, indicating accurate and reliable classification results. Notably, labels 31281, 43210, and 78351 exhibited high precision, recall, F1-score, and Jaccard score values, suggesting excellent performance for these specific biological activities.

However, labels 71522, 78348, and 12464 presented lower F1-scores and Jaccard scores compared to other labels, indicating room for improvement in the agent's classification performance for these specific activities. Further analysis and refinement may be necessary to enhance the agent's accuracy and consistency in multi-label classification for these tasks.

\begin{figure}[H]
  \centering
  \begin{subfigure}[b]{\textwidth}
    \centering
    \includegraphics[width=0.46\textwidth]{Images/chap4/DS1 heatmap.pdf}
    \caption{DS1}
  \end{subfigure}
  \begin{subfigure}[b]{0.46\textwidth}
    \includegraphics[width=\textwidth]{Images/chap4/DS2 heatmap.pdf}
    \caption{DS2}
  \end{subfigure}
  \begin{subfigure}[b]{0.45\textwidth}
    \includegraphics[width=\textwidth]{Images/chap4/DS3 heatmap.pdf}
    \caption{DS3}
  \end{subfigure}
 
  \caption{Classification test report}
  \label{fig:class_report}
\end{figure}


The classification test results provide insights into the performance of the PPO agent trained for multi-label classification of biological activities. The evaluation metrics used to assess the agent's performance were recall, precision, F1-score, Jaccard score, and hamming loss. The micro-average and macro-average scores, along with the hamming loss, are discussed below.

Micro-average scores represent an overall performance measure that considers the total true positives, false positives, and false negatives across all labels. The micro-average recall score of 0.602 suggests that, on average, the model is able to correctly identify approximately 60.2\% of the relevant instances. This indicates room for improvement in capturing a higher proportion of the relevant instances. The micro-average precision score of 0.983 indicates a high level of accuracy in the model's predictions, with only 1.7\% of the positive predictions being false positives. The micro-average F1-score of 0.747 reflects a balance between precision and recall. The micro-average Jaccard score of 0.597 represents the similarity between the predicted and true label sets, with a higher value indicating a closer match. The relatively low micro-average values suggest the need for further refinement to improve recall and achieve a better balance between precision and recall.

Macro-average scores calculate the average performance across all labels independently, without considering class imbalance. The macro-average recall score of 0.671 indicates that, on average, the model performs well in capturing the relevant instances for each individual label. This suggests that the model shows good performance for certain specific labels. The macro-average precision score of 0.981 suggests a high level of precision in the model's predictions across all labels. The macro-average F1-score of 0.768 represents the harmonic mean of precision and recall, indicating a good balance between the two metrics. The macro-average Jaccard score of 0.662 reflects the average similarity between the predicted and true label sets for each individual label. These values indicate that the model performs well on average for the individual labels, although there may be variation in performance across different labels.

The hamming loss, with a value of 0.023, indicates a relatively low error rate in the model's predictions. This suggests that the model assigns labels correctly to the majority of instances.

## LBVS Results

To test the effectiveness of the PPO-JAC model in producing a molecular representation that accurately captures the features of the activity classes and distinguishes between active molecules and decoys, we evaluated its performance using the MDDR dataset. Specifically, we conducted a standard LBVS performance test and calculated recall values for each activity class to assess how well our model performed compared to other methods. These recall values serve as a crucial metric for evaluating model performance.


To simulate the LBVS process the following steps were implemented:

1- Ten reference structures were selected from each activity class of MDDR-DS1, MDDR-DS2, and MDDR-DS3.

2- Similarity calculations were carried out between the reference structures of each activity class and all molecules in the databases.

3- The outcomes of these calculations were ranked in descending order, and only the top 1\% and 5\% were picked for each reference structure.

4- The retrieval outcomes for each reference structure were examined to determine the number of active molecules that fell under the same activity category, known as true positive values(recall).

5- The recall value for the activity class was determined by computing the average of values obtained from ten reference structures. These values were calculated for both 1\% and 5\% cutoffs, and this process was repeated for all datasets.

The results obtained by the simulation of the LBVS process on the MDDR dataset (DS1, DS2, and DS3) are organized in tables as follows:

The \cref{tab:chapter4.1,tab:chapter4.3,tab:chapter4.5} present a comprehensive overview of obtained results of recall values for each of the activity classes in the dataset, where the first column lists the corresponding categories, and the succeeding columns report the average recall values for all activity classes at a cut-off of 1\%. Similarly, \cref{tab:chapter4.2,tab:chapter4.4,tab:chapter4.6} show the same information but with a cut-off of 5\%.

The overall average recall results for all classes are shown at the end of each column, with the best average recall for each activity class being highlighted. Additionally, at the bottom of each column, a row of shaded cells is presented, which corresponds to the total number of shaded cells for all the similarity methods that achieved the best results.


The study involved a comparison of the proposed PPO-JAC approach with five existing state-of-the-art benchmark methods. These existing methods included the Bayesian inference Network method (BIN)\cite{abdo2009similarity}, Tanimoto method (TAN)\cite{noauthor_why_nodate}, Stack of Deep Belief Networks method (SDBN)\cite{nasser2020improved}, Deep Convolutional Neural Network method (DCNNLB)\cite{berrhail_deep_2022}, Hybrid-Enhanced Siamese Similarity method (Hybrid-F-Max)\cite{altalib_hybrid-enhanced_2022}. The evaluation was conducted using the ANOVA test to determine which approach had the best screening effectiveness on all MDDR datasets. To determine an overall ranking for these similarity methods, average recall values were calculated for all activity classes in MDDR. The Top 1\% and Top 5\% screening recall values for all data sets in MDDR were then compared across all similarity methods, and results are depicted in \cref{fig:models_boxplot_ds1.1,fig:models_boxplot_ds1.5,fig:models_boxplot_ds2.1,fig:models_boxplot_ds2.5,fig:models_boxplot_ds3.1,fig:models_boxplot_ds3.5}.

### DS1 Results

In Table \ref{tab:chapter4.1} and \ref{tab:chapter4.2}, we recorded the recall values for the 1\% and 5\% cutoffs of MDDR-DS1(structurally homogeneous and heterogeneous).

Regarding the top 1\%, our method achieved higher recall values than all other methods in 10 out of 11 activity classes. However, in the activity class  "31432" the Hybrid-F-Max method outperformed our method by a recall difference of 7.51.

For the top 5\%, our proposed PPO-JAC model scored the highest recall value for 7 out of 11 activity classes. However, it was outperformed by the Hybrid-F-Max method in the activity classes "31420", "37110", and "31432" by recall differences of 5.06, 8.14, and 6.32, respectively. Additionally, it was outperformed by the DCNNLB method in the activity classes "31420" and "31432" by recall differences of 1.12 and 8.47, respectively. It was also slightly outperformed by the SDBN method in the activity class "31420" by a recall difference of 0.3.

Based on the mean recall and the number of shaded cells values in tables \ref{tab:chapter4.1} and \ref{tab:chapter4.2}, the PPO-JAC model had the best retrieval recall results in the MDDR-DS1 dataset, followed by the Hybrid-F-Max method, DCNNLB, SDBN, BIN, and TAN, respectively.


\begin{table}[H]
\begin{center}
\resizebox{0.8\linewidth}{!}{
\begin{tabular}{c|ccccc|c}
\hline \hline
\multicolumn{7}{c}{Top 1\%}\\
\hline
\rowcolor{header}
&\multicolumn{5}{c|}{ \color{header_text}Previous Methods} & \color{header_text}Our Proposed method\\
\rowcolor{header}
\color{header_text}Activity index & \color{header_text}BIN   & \color{header_text}TAN   & \color{header_text}DCNNLB & \color{header_text}SDBN  & \color{header_text}Hybrid-F-Max & \textbf{\color{header_text}PPO-JAC} \\
\hline \hline
31420          & 74.08 & 69.69 & 80.8   & 74.21 & 88.28        & \cellcolor{highlight}86.2    \\
71523          & 28.26 & 25.94 & 59.92  & 27.97 & 66.45        & \cellcolor{highlight}77.5    \\
37110          & 26.05 & 9.63  & 39.65  & 26.03 & 58.13        & \cellcolor{highlight}60.4    \\
31432          & 39.23 & 35.82 & 45.12  & 39.79 & \cellcolor{highlight}69.72        & 47.3    \\
42731          & 21.68 & 17.77 & 48.6   & 23.06 & 55.45        & \cellcolor{highlight}80      \\
6233           & 14.06 & 13.87 & 35.25  & 19.29 & 47.93        & \cellcolor{highlight}64.6    \\
6245           & 6.31  & 6.51  & 21.64  & 6.27  & 16.25        & \cellcolor{highlight}57.1    \\
7701           & 11.45 & 8.63  & 19.95  & 14.05 & 22.03        & \cellcolor{highlight}49.1    \\
6235           & 10.84 & 9.71  & 21.55  & 12.87 & 23.58        & \cellcolor{highlight}54.8    \\
78374          & 14.25 & 13.69 & 29.62  & 17.47 & 25.89        & \cellcolor{highlight}68      \\
78331          & 6.03  & 7.17  & 13.63  & 9.93  & 14           & \cellcolor{highlight}78.6    \\
\hline\hline
Mean           & 22.93 & 19.86 & 37.79  & 24.63 & 44.34        & \color{header_text}\cellcolor{highlight2}65.78   \\
%\hline
%Median         & 14.25  & 13.69 & 35.25 & 19.29 & 47.93       & \color{header_text}\cellcolor{highlight2}64.6   \\
\hline
Shaded cells   & 0     & 0     & 0      & 0     & 1            & \color{header_text}\cellcolor{highlight2}10        \\
\hline\hline
\end{tabular}
}
\caption{The obtained results of recall values in top 1\% for the DS1 dataset.}
\label{tab:chapter4.2}
\end{center}
\end{table}


\begin{table}[H]
\begin{center}
\resizebox{0.8\linewidth}{!}{
\begin{tabular}{c|ccccc|c}
\hline\hline
\multicolumn{7}{c}{Top 5\%}\\
\hline
\rowcolor{header}
&\multicolumn{5}{c|}{ \color{header_text}Previous Methods} & \color{header_text}Our Proposed method\\
\rowcolor{header}
\color{header_text}Activity index & \color{header_text}BIN   & \color{header_text}TAN   & \color{header_text}DCNNLB & \color{header_text}SDBN  & \color{header_text}Hybrid-F-Max & \textbf{\color{header_text}PPO-JAC} \\
\hline \hline
31420          & 87.61 & 83.49 & 90.12  & 89.03 & \cellcolor{highlight}94.06        & 89      \\
71523          & 52.72 & 48.92 & 86.31  & 65.17 & 86.44        & \cellcolor{highlight}92      \\
37110          & 48.2  & 21.01 & 75.95  & 41.25 & \cellcolor{highlight}84.54        & 76.4    \\
31432          & 77.57 & 74.29 & \cellcolor{highlight}97.17  & 79.87 & 95.02        & 88.7    \\
42731          & 26.63 & 29.68 & 72.32  & 31.92 & 77.27        & \cellcolor{highlight}97.3    \\
6233           & 23.49 & 27.68 & 67.22  & 29.31 & 80.2         & \cellcolor{highlight}85.8    \\
6245           & 14.86 & 16.54 & 52.09  & 21.06 & 39.35        & \cellcolor{highlight}88.8    \\
7701           & 27.79 & 24.09 & 52.51  & 28.43 & 49.65        & \cellcolor{highlight}75.1    \\
6235           & 23.78 & 20.06 & 50.41  & 27.82 & 53.21        & \cellcolor{highlight}86.7    \\
78374          & 20.2  & 20.51 & 46.16  & 19.09 & 57.82        & \cellcolor{highlight}80.8    \\
78331          & 11.8  & 16.2  & 37.64  & 16.21 & 39.69        & \cellcolor{highlight}85.5    \\
\hline\hline
Mean           & 37.7  & 34.77 & 66.17  & 40.83 & 68.84        & \color{header_text}\cellcolor{highlight2}86.01   \\
%\hline
%Median         & 26.63  & 24.09 & 67.22 & 29.31 & 77.27       & \color{header_text}\cellcolor{highlight2}86.7   \\
\hline
Shaded cells   & 0     & 0     & 1      & 0     & 2            & \color{header_text}\cellcolor{highlight2}8        \\
\hline\hline
\end{tabular}
}
\caption{The obtained results of recall values in top 5\% for the DS1 dataset.}
\label{tab:chapter4.3}
\end{center}
\end{table}


\begin{figure}[H]
    \centering
    \includegraphics[scale=0.35]{Images/chap4/ds1 1.pdf}
    \caption{Comparison of the performance values (recall) of similarity methods for DS1 dataset in Top 1\% using ANOVA.}
    \label{fig:models_boxplot_ds1.1}
\end{figure}


The comparison of the screening recall values obtained in data set DS1 for the Top 1\%, which is shown in \ref{fig:models_boxplot_ds1.1}, showed that our proposed PPO-JAC method has the best recall value with a median of: 64.6. Therefore, the ranking of the six similarity methods was:

PPO-JAC > Hybrid-F-Max(47.93) > DCNNLB(35.25) > SDBN(19.29) > BIN(14.25) > TAN(13.69).

The results obtained in the 5\% cut-off for the Anova test analysis for the DS1 data set are shown in \ref{fig:models_boxplot_ds1.5}. This latter indicates, that our method gives the best mean recall value with a median: of 86.7 which shows a significant difference in the performance of our method. Thus, the overall ranking of similarity methods is as follows:

PPO-JAC > Hybrid-F-Max(77.27) > DCNNLB(67.22) > SDBN(29.31) > BIN(26.63) > TAN(24.09).


\begin{figure}[H]
    \centering
    \includegraphics[scale=0.35]{Images/chap4/ds1 5.pdf}
    \caption{Comparison of the performance values (recall) of similarity methods for DS1 dataset in Top 5\% using ANOVA.}
    \label{fig:models_boxplot_ds1.5}
\end{figure}


### DS2 Results

The MDDR-DS2(structurally homogeneous) dataset's top 1\% and top 5\% recall values for our method and various previous methods of the state-of-art are presented in the tables below \cref{tab:chapter4.3,tab:chapter4.4}. As we can see the proposed method performed better than most previous methods.

Regarding the top 1\%, our method had the highest recall value in most cases, except for three activity indexes where it was outperformed by the methods Hybrid-F-Max and SDBN in the activity index "64100" by recall differences of 1.32 and 1.75, respectively. Additionally, it was outperformed by the methods DCNNLB and Hybrid-F-Max in the activity index "64220" by recall differences of 1.19 and 11.23, respectively. And also it was outperformed by the methods SDBN, BIN, and DCNNLB in the activity index "75755" by recall differences of 0.12, 1.66, and 2.06, respectively.

For the top 5\%, our proposed PPO-JAC model scored the highest recall value for 7 out of 10 activity classes. However, it was outperformed only in two activity indexes. The first one was only by the method SDBN in the activity index "64220" with a recall difference of 0.24. And the second one was by the methods BIN and DCNNLB in the activity index "75755" by recall differences of 1.65 and 2.97, respectively.

Overall, regardless of the lack of performance of the model in some activity classes, PPO-JAC outperforms all previous approaches based on their respective mean recall values in both top percentages in the MDDR-DS2 dataset.

\begin{table}[H]
\begin{center}
\resizebox{0.8\linewidth}{!}{
\begin{tabular}{c|ccccc|c}
\hline\hline
\multicolumn{7}{c}{Top 1\%}\\
\hline
\rowcolor{header}
&\multicolumn{5}{c|}{ \color{header_text}Previous Methods} & \color{header_text}Our Proposed method\\
\rowcolor{header}
\color{header_text}Activity index & \color{header_text}BIN   & \color{header_text}TAN   & \color{header_text}DCNNLB & \color{header_text}SDBN  & \color{header_text}Hybrid-F-Max & \textbf{\color{header_text}PPO-JAC} \\
\hline \hline
7707           & 72.18 & 61.84 & 73.86  & 83.19 & 83.61        & \cellcolor{highlight}99      \\
7708           & 96    & 47.03 & 83.85  & 94.82 & 93.1         & \cellcolor{highlight}98.7    \\
31420          & 79.82 & 65.1  & 65.97  & 79.27 & 77.35        & \cellcolor{highlight}95.9    \\
42710          & 76.27 & 81.27 & 92.07  & 74.81 & 84.91        & \cellcolor{highlight}99.9    \\
64100          & 88.43 & 80.31 & 87.32  & \cellcolor{highlight}93.65 & 93.22        & 91.9    \\
64200          & 70.18 & 53.84 & 76.65  & 71.16 & 60.39        & \cellcolor{highlight}99.2    \\
64220          & 68.32 & 38.64 & 82.39  & 68.71 & \cellcolor{highlight}92.34        & 81.2    \\
64500          & 81.2  & 30.56 & 87.94  & 75.62 & 73.2         & \cellcolor{highlight}99.2    \\
64350          & 81.89 & 80.18 & 75.41  & 85.21 & 85.06        & \cellcolor{highlight}95.1    \\
75755          & 98.06 & 87.56 & \cellcolor{highlight}98.46  & 96.52 & 90.99        & 96.4    \\
\hline\hline
Mean           & 81.24 & 62.63 & 82.39  & 82.3  & 83.42        & \color{header_text}\cellcolor{highlight2}95.65   \\
%\hline
%Median         & 80.51  & 63.47 & 83.12 & 81.23 & 84.99       & \color{header_text}\cellcolor{highlight2}97.55   \\
\hline
Shaded cells   & 0     & 0     & 0      & 2     & 1            & \color{header_text}\cellcolor{highlight2}7            \\
\hline\hline
\end{tabular}
}
\caption{The obtained results of recall values in top 1\% for the DS2 dataset.}
\label{tab:chapter4.4}
\end{center}
\end{table}



\begin{table}[H]
\begin{center}
\resizebox{0.8\linewidth}{!}{
\begin{tabular}{c|ccccc|c}
\hline\hline
\multicolumn{7}{c}{Top 5\%}\\
\hline
\rowcolor{header}
&\multicolumn{5}{c|}{ \color{header_text}Previous Methods} & \color{header_text}Our Proposed method\\
\rowcolor{header}
\color{header_text}Activity index & \color{header_text}BIN   & \color{header_text}TAN   & \color{header_text}DCNNLB & \color{header_text}SDBN  & \color{header_text}Hybrid-F-Max & \textbf{\color{header_text}PPO-JAC} \\
\hline \hline
7707           & 74.81 & 70.39 & 90.53  & 73.9  & 87.17        & \cellcolor{highlight}99.5    \\
7708           & 99.61 & 56.58 & 95.51  & 98.22 & 93.94        & \cellcolor{highlight}99.8    \\
31420          & 65.46 & 88.19 & 94.03  & 95.64 & 94.89        & \cellcolor{highlight}98.8    \\
42710          & 92.55 & 88.09 & 98.29  & 90.12 & 85.64        & \cellcolor{highlight}100     \\
64100          & 99.22 & 93.75 & 98.62  & 99.05 & 96.41        & \cellcolor{highlight}99.9    \\
64200          & 99.2  & 77.68 & 91.9   & 93.76 & 70.58        & \cellcolor{highlight}100     \\
64220          & 91.32 & 52.19 & \cellcolor{highlight}99.64  & 96.01 & 94.28        & 99.4    \\
64500          & 94.96 & 44.8  & 98.89  & 91.51 & 74.16        & \cellcolor{highlight}100     \\
64350          & 91.47 & 91.71 & 99.28  & 86.94 & 90.65        & \cellcolor{highlight}97.5    \\
75755          & 98.35 & 94.82 & \cellcolor{highlight}99.67  & 91.6  & 90.99        & 96.7    \\
\hline\hline
Mean           & 90.7  & 75.82 & 96.64  & 91.68 & 87.87        & \color{header_text}\cellcolor{highlight2}99.16   \\
%\hline
%Median         & 93.76  & 82.89 & 98.46 & 92.68 & 90.82       & \color{header_text}\cellcolor{highlight2}99.65   \\
\hline
Shaded cells   & 0     & 0     & 2      & 0     & 0            & \color{header_text}\cellcolor{highlight2}8            \\
\hline\hline
\end{tabular}
}
\caption{The obtained results of recall values in top 5\% for the DS2 dataset.}
\label{tab:chapter4.5}
\end{center}
\end{table}


\begin{figure}[H]
    \centering
    \includegraphics[scale=0.35]{Images/chap4/ds2 1.pdf}
    \caption{Comparison of the performance values (recall) of similarity methods for DS2 dataset in Top 1\% using ANOVA.}
    \label{fig:models_boxplot_ds2.1}
\end{figure}


In \ref{fig:models_boxplot_ds2.1}, we present the results obtained by the Anova test for the DS2 data set in the Top 1\%. A median value of 97.55 for the PPO-JAC method shows superior performance compared to the other methods. Thus, the overall ranking of the six methods is as follows:

PPO-JAC > Hybrid-F-Max(84.99) > DCNNLB(83.12) > SDBN(81.23) > BIN(80.51) > TAN(63.47).

In addition, the results obtained from the ANOVA test in DS2 for Top 5\% \ref{fig:models_boxplot_ds2.5} showed that the proposed PPO-JAC method has a higher median value than all other methods (99.65), then 93.76 for BIN, 82.89 for TAN, 98.46 for DCNNLB, 92.68 for SDBN and 90.82 for Hybrid-F-Max. Thus, the overall ranking of the six similarity methods is as follows:

PPO-JAC > DCNNLB > BIN > SDBN > Hybrid-F-Max > TAN.

\begin{figure}[H]
    \centering
    \includegraphics[scale=0.35]{Images/chap4/ds2 5.pdf}
    \caption{Comparison of the performance values (recall) of similarity methods for DS2 dataset in Top 5\% using ANOVA.}
    \label{fig:models_boxplot_ds2.5}
\end{figure}

### DS3 Results

The MDDR-DS3 (structurally heterogeneous) recall values for the 1\% and 5\% cut-offs recorded in the tables below \cref{tab:chapter4.5,tab:chapter4.6} demonstrated that the proposed PPO-JAC similarity models were superior to the benchmark previous studies by far, especially in the 1\% cut-off were our method scored a 10 out of 10 performance surpassing all the previous methods in all activity indexes by a mean recall of 63.07 with a difference of 29.78 from the best state-of-art method in the MDDR-DS3(Hybrid-F-Max) which is considered as a decent improvement acknowledging the heterogeneous nature of the MDDR-DS3.

Additionally, for the top 5\%, our proposed PPO-JAC model scored the highest recall value for 6 out of 10 activity classes. And was outperformed in four activity indexes. The first one was only by the method DCNNLB in the activity index "9249" with a recall difference of 15.1. And the second one was by the methods DCNNLB and Hybrid-F-Max in the activity index "31281" by recall differences of 5.57 and 8.87, respectively. And the third one was only by the method DCNNLB in the activity index "71522" with a recall difference of 12.04. And the fourth one was only by the method Hybrid-F-Max in the activity index "75721" with a recall difference of 5.46.

The proposed method, PPO-JAC, outperformed all the previous methods in both the top 1\% and the top 5\% of the DS3 dataset, indicating its effectiveness in recognizing biological activities overtaking the heterogeneous nature of the MDDR-DS3 dataset.



\begin{table}[H]
\begin{center}
\resizebox{0.8\linewidth}{!}{
\begin{tabular}{c|ccccc|c}
\hline\hline
\multicolumn{7}{c}{Top 1\%}\\
\hline
\rowcolor{header}
&\multicolumn{5}{c|}{ \color{header_text}Previous Methods} & \color{header_text}Our Proposed method\\
\rowcolor{header}
\color{header_text}Activity index & \color{header_text}BIN   & \color{header_text}TAN   & \color{header_text}DCNNLB & \color{header_text}SDBN  & \color{header_text}Hybrid-F-Max & \textbf{\color{header_text}PPO-JAC} \\
\hline \hline
9249           & 15.33 & 12.12 & 74.71  & 19.47 & 42.19        & \cellcolor{highlight}68.2    \\
12455          & 9.37  & 6.57  & 28.08  & 13.29 & 23.26        & \cellcolor{highlight}57.1    \\
12464          & 8.45  & 8.17  & 20.95  & 12.91 & 41.33        & \cellcolor{highlight}87.4    \\
31281          & 18.29 & 16.95 & 36.89  & 23.62 & 68.38        & \cellcolor{highlight}76.5    \\
43210          & 7.34  & 6.27  & 41.53  & 14.23 & 37.35        & \cellcolor{highlight}47.5    \\
71522          & 4.08  & 3.75  & 52.53  & 11.92 & 15.43        & \cellcolor{highlight}47.2    \\
75721          & 20.41 & 17.32 & 26.38  & 29.08 & 56.06        & \cellcolor{highlight}63.4    \\
78331          & 7.51  & 6.31  & 27.56  & 11.93 & 16.65        & \cellcolor{highlight}78.6    \\
78348          & 9.79  & 10.15 & 17.42  & 9.17  & 15.79        & \cellcolor{highlight}36.3    \\
78351          & 13.68 & 9.84  & 11.52  & 18.13 & 16.42        & \cellcolor{highlight}68.5    \\
\hline\hline
Mean           & 11.43 & 9.75  & 33.76  & 16.38 & 33.29        & \color{header_text}\cellcolor{highlight2}63.07   \\
%\hline
%Median         & 9.58  & 9.01 & 27.82 & 13.76 & 30.31       & \color{header_text}\cellcolor{highlight2}65.8   \\
\hline
Shaded cells   & 0     & 0     & 0      & 0     & 0            & \color{header_text}\cellcolor{highlight2}10     \\
\hline\hline
\end{tabular}
}
\caption{The obtained results of recall values in top 1\% for the DS3 dataset.}
\label{tab:chapter4.6}
\end{center}
\end{table}

\begin{table}[H]
\begin{center}
\resizebox{0.8\linewidth}{!}{
\begin{tabular}{c|ccccc|c}
\hline\hline
\multicolumn{7}{c}{Top 5\%}\\
\hline
\rowcolor{header}
&\multicolumn{5}{c|}{ \color{header_text}Previous Methods} & \color{header_text}Our Proposed method\\
\rowcolor{header}
\color{header_text}Activity index & \color{header_text}BIN   & \color{header_text}TAN   & \color{header_text}DCNNLB & \color{header_text}SDBN  & \color{header_text}Hybrid-F-Max & \textbf{\color{header_text}PPO-JAC} \\
\hline\hline
9249           & 25.72 & 24.17 & \cellcolor{highlight}98.3   & 31.61 & 67.78        & 83.2    \\
12455          & 14.65 & 10.29 & 58.84  & 16.29 & 47.07        & \cellcolor{highlight}83.9    \\
12464          & 16.55 & 15.22 & 47.74  & 20.9  & 80.2         & \cellcolor{highlight}92.3    \\
31281          & 28.29 & 29.62 & 85.47  & 36.13 & \cellcolor{highlight}88.57        & 79.7    \\
43210          & 14.41 & 16.07 & 73.75  & 22.09 & 51.15        & \cellcolor{highlight}79.4    \\
71522          & 8.44  & 12.37 & \cellcolor{highlight}86.54  & 14.68 & 31.36        & 74.5    \\
75721          & 30.02 & 25.21 & 50     & 41.07 & \cellcolor{highlight}98.66        & 93.2    \\
78331          & 12.03 & 15.01 & 52.08  & 17.13 & 42.36        & \cellcolor{highlight}85.3    \\
78348          & 20.76 & 24.67 & 50.18  & 26.93 & 47.8         & \cellcolor{highlight}66.4    \\
78351          & 12.94 & 11.71 & 16.49  & 17.87 & 37.89        & \cellcolor{highlight}83.5    \\
\hline\hline
Mean           & 18.38 & 18.43 & 61.94  & 24.47 & 59.28        & \color{header_text}\cellcolor{highlight2}82.14   \\
%\hline
%Median         & 15.6  & 15.65 & 55.46 & 21.5 & 49.48       & \color{header_text}\cellcolor{highlight2}83.35   \\
\hline
Shaded cells   & 0     & 0     & 2      & 0     & 2            & \color{header_text}\cellcolor{highlight2}6          \\
\hline\hline
\end{tabular}
}
\caption{The obtained results of recall values in top 5\% for the DS3 dataset.}
\label{tab:chapter4.7}
\end{center}
\end{table}


\begin{figure}[H]
    \centering
    \includegraphics[scale=0.35]{Images/chap4/ds3 1.pdf}
    \caption{Comparison of the performance values (recall) of similarity methods for DS3 dataset in Top 1\% using ANOVA.}
    \label{fig:models_boxplot_ds3.1}
\end{figure}


The \ref{fig:models_boxplot_ds3.1} shows the screening recall results obtained by ANOVA test for MDDR DS3 at Top 1\%. These results showed superior performance of our method, with a mean recall equal to 63.07 which means a significant difference in performance between similarity methods. The median values for the BIN, TAN, DCNNLB, SDBN, Hybrid-F-Max, and PPO-JAC methods were: 9.58, 9.01, 27.82, 13.76, 30.31, and 65.8, respectively. Thus, the overall ranking of similarity methods is as follows:

PPO-JAC > Hybrid-F-Max > DCNNLB > SDBN > BIN > TAN.

The comparison shown in \ref{fig:models_boxplot_ds3.5} of the screening recall values obtained in DS3, at top 5\% demonstrated, that the PPO-JAC method presents a superiority in performance with mean recall equal to 82.14 (a significant difference in performance), and the median values were found at : 15.6, 15.65, 55.46, 21.5, 49.48, and 83.35 for the BIN, TAN, DCNNLB, SDBN, Hybrid-F-Max, and PPO-JAC respectively. As a result, the overall ranking of similarity methods is as follows:

PPO-JAC > DCNNLB > Hybrid-F-Max > SDBN > TAN > BIN.

\begin{figure}[H]
    \centering
    \includegraphics[scale=0.35]{Images/chap4/ds3 5.pdf}
    \caption{Comparison of the performance values (recall) of similarity methods for DS3 dataset in Top 5\% using ANOVA.}
    \label{fig:models_boxplot_ds3.5}
\end{figure}

