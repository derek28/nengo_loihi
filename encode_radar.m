%% Convert the spike timestamp file to SLAYER compatible format
n_total = 2400;
n_train = 1920;
n_test = 480;
% 
% n_total = 10;
% n_train = 8;
% n_test = 2;

% label information files
train_file_name = "gesture_lsm003_slayer/train1920.txt";
fout_train = fopen(train_file_name, 'w');

test_file_name = "gesture_lsm003_slayer/test480.txt";
fout_test = fopen(test_file_name, 'w');

fprintf(fout_train, "#sample\t#class\n");
fprintf(fout_test, "#sample\t#class\n");

% timestamp files, training
for ii = 1:n_train
    in_file_name = sprintf("gesture_lsm_003/gesture_%d.txt", ii);
    out_file_name = sprintf("gesture_lsm003_slayer/%d.bs1", ii);
    fin = fopen(in_file_name, 'r');
    
    label = str2num(fgetl(fin)) - 1;
    
    neuronID = [];
    timeID = [];
    polarity = [];
    
    while ~feof(fin)
        line = fgetl(fin);
    
        x = str2num(line);
        id = round(x(1) / 2);  % (1, 2), (3, 4), (5, 6), ...
        time = x(2);
        pol = mod(x(1), 2);       % 1, 3, 5 positive; 2, 4, 6 negative. 
        
        neuronID = [neuronID; id];
        timeID = [timeID; time];
        polarity = [polarity; pol];
    
        %fprintf("%d %d %0.4f\n", id, pol, time);
    end
        
    fclose(fin);

    % write the label information file. Label starts from 0. 
    fprintf(fout_train, "%d\t%d\n", ii, label);
    
    % neuronID starts from 1.  { neuronID -= 1 } in the encoding function. 
    eventStamp = [neuronID, timeID, polarity];
    
    % encodeSpikes(<name fo spike data file>, <eventStamp = [neuronID, timeID(ms), polarity]>)
    encode1DBinSpikes(out_file_name, eventStamp);
end

fclose(fout_train);

% timestamp files, testing
for ii = (n_train + 1):n_total
    in_file_name = sprintf("gesture_lsm_003/gesture_%d.txt", ii);
    out_file_name = sprintf("gesture_lsm003_slayer/%d.bs1", ii);
    fin = fopen(in_file_name, 'r');

    % from 0 to 11. total 12 labels. 
    label = str2num(fgetl(fin)) - 1;
    
    neuronID = [];
    timeID = [];
    polarity = [];
    
    while ~feof(fin)
        line = fgetl(fin);
    
        x = str2num(line);
        id = round(x(1) / 2);  % (1, 2), (3, 4), (5, 6), ...
        time = x(2);
        pol = mod(x(1), 2);       % 1, 3, 5 positive; 2, 4, 6 negative. 
        
        neuronID = [neuronID; id];
        timeID = [timeID; time];
        polarity = [polarity; pol];
    
        %fprintf("%d %d %0.4f\n", id, pol, time);
    end
        
    fclose(fin);

    % write the label information file. Label starts from 0. 
    fprintf(fout_test, "%d\t%d\n", ii, label);
    
    eventStamp = [neuronID, timeID, polarity];
    
    % encodeSpikes(<name fo spike data file>, <eventStamp = [neuronID, timeID(ms), polarity]>)
    encode1DBinSpikes(out_file_name, eventStamp);
end

fclose(fout_test);