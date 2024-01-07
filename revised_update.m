% Read the file
data = readtable('networkdata.xlsx');

% Define your condition for the new 'DX' column here
% In this example, if 'DX_3' column value is equal to 1, set 'DX' to 0; otherwise, set 'DX' to 1
data.DX = (data.DX_3 == 1) * 1 + (data.DX_3 ~= 1) * 0;

% Write the updated table back to an Excel file
writetable(data, 'updated_networkdata.xlsx');
