--[[ 
    -- Synthetic data for following tasks
    -- task1: repeat each symbol twice. e.g. 1234 -> 11223344
    -- task2: append last element of the input sequence to itself. e.g. 123 -> 1233

    -- Alert: ugly code. 
    -- Excuse: I'm new to lua/torch and in a hurry.
]]--

local Synthetic = {}
Synthetic.__index = Synthetic

-- helpers to filter out already generated data

Buffer = {}
function alreadyGenerated(new)
    for _, old in pairs(Buffer) do
        if #old == #new and torch.all(old:eq(new)) then return true end
    end
    return false
end

function buffer(tensor)
    --table.insert(Buffer, torch.Tensor(table))
    Buffer[#Buffer + 1] = tensor
end
--


-- task: repeat each symbol twice. e.g. 1234 -> 11223344
function Synthetic.generate1(vocab_size, minSeqLength, maxSeqLength)
    -- generate parameters for sequence generation randomly
    local seqLen = math.random(minSeqLength, maxSeqLength)

    -- generate sequences
    local encIn = torch.Tensor(seqLen)
    local decIn = torch.Tensor(2*seqLen+1)
    local decOut = torch.Tensor(2*seqLen+1)
    
    -- check if this sequence has already been generated, if yes try again else go ahead
    repeat
        encIn:apply( function() return math.random(vocab_size - 2) end)
    until not alreadyGenerated(encIn)

    buffer(encIn)

    decIn[1] = vocab_size-1                 -- GO
    decOut[2*seqLen+1] = vocab_size         -- EOS
    i = 0
    decIn:sub(2,2*seqLen+1):apply(function()  i = i+1; if i%2 ~= 0 then return encIn[(i+1)/2] else return encIn[i/2] end end)
    i = 0
    decOut:sub(1,2*seqLen):apply(function()  i = i+1; if i%2 ~= 0 then return encIn[(i+1)/2] else return encIn[i/2] end end)
    return {encIn, decIn, decOut}
end


-- task: append last element of the sequence to the input seq. e.g. 123 -> 1233
function Synthetic.generate2(vocab_size, minSeqLength, maxSeqLength)
    -- generate parameters for sequence generation randomly
    local seqLen = math.random(minSeqLength, maxSeqLength)

    -- generate sequences
    local encIn = torch.Tensor(seqLen)
    local decIn = torch.Tensor(seqLen+2)
    local decOut = torch.Tensor(seqLen+2)
    
    -- check if this sequence has already been generated, if yes try again else go ahead
    repeat
        encIn:apply( function() return math.random(vocab_size - 2) end)
    until not alreadyGenerated(encIn)

    buffer(encIn)

    decIn[1] = vocab_size-1            -- GO
    decIn:sub(2,seqLen+1):copy(encIn)
    decIn[seqLen+2] = decIn[seqLen+1]

    decOut[seqLen+2] = vocab_size      -- EOS
    decOut:sub(1,seqLen):copy(encIn)
    decOut[seqLen+1] = decOut[seqLen]

    return {encIn, decIn, decOut}
end


function Synthetic.create(which, dataSize, batch_size, train_frac, val_frac, test_frac)
    local self = {}
    setmetatable(self, Synthetic)

    -- required to generate data
    self.vocab_size = 6
    self.minSeqLength = 2
    self.maxSeqLength = 5
    self.vocab_mapping = {}
    generate = which == 1 and self.generate1 or which == 2 and self.generate2
    fileName = 'synthetic_data-' .. tostring(which) .. '.t7'

    self.ntrain = math.floor(dataSize * train_frac)
    self.nval = math.floor(dataSize * val_frac)
    self.ntest = dataSize - (self.ntrain + self.nval)

    for i = 1, self.vocab_size do self.vocab_mapping[tostring(i)] = i end

    if not path.exists(fileName) then
        print('Generating a new data set...')
        self.train_data = {}
        self.val_data = {}
        self.test_data = {}

        for i = 1, self.ntrain do
            gen_data = generate(self.vocab_size, self.minSeqLength, self.maxSeqLength)
            if not gen_data then print('generated_data:', gen_data) end
            table.insert(self.train_data, gen_data)
        end

        for i = 1, self.nval do
            gen_data = generate(self.vocab_size, self.minSeqLength, self.maxSeqLength)
            if not gen_data then print('generated_data:', gen_data) end
            table.insert(self.val_data, gen_data)
        end

        for i = 1, self.ntest do
            gen_data = generate(self.vocab_size, self.minSeqLength, self.maxSeqLength)
            if not gen_data then print('generated_data:', gen_data) end
            table.insert(self.test_data, gen_data)
        end
                
        torch.save(fileName, {self.train_data, self.val_data, self.test_data, self.ntrain, self.nval, self.ntest})
    else
        print('Using existing data set...')
        self.train_data, self.val_data, self.test_data, self.ntrain, self.nval, self.ntest = unpack(torch.load(fileName))
    end

    self.train_index = 0
    self.val_index = 0
    self.test_index = 0
    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_size = batch_size
    self.train_batch = {}
    self.val_batch = {}
    self.test_batch = {}
    
    if self.batch_size > 1 then
        for splitI = 1, 3 do self:createBatches(splitI) end
        self.ntrain, self.nval, self.ntest = #(self.train_batch), #(self.val_batch), #(self.test_batch)
    else
        self.train_batch, self.val_batch, self.test_batch = self.train_data, self.val_data, self.test_data
    end

    self.batch_split_sizes = {#(self.train_batch), #(self.val_batch), #(self.test_batch)}

    print('generated data for task:', which, 'split_size:', self.batch_split_sizes)
    return self
end


-- summary: it takes similar length sequences and arrange them in batches.
function Synthetic:createBatches(split_index)
    local n, batch, data
    if split_index == 1 then
        batch = self.train_batch; n = self.ntrain; data = self.train_data
    elseif split_index == 2 then
        batch = self.val_batch; n = self.nval; data = self.val_data
    elseif split_index == 3 then
        batch = self.test_batch; n = self.ntest; data = self.test_data
    end

    local cat = torch.cat
    local bs = self.batch_size
    local encSeq, decInSeq, decOutSeq, seqLen
    local encGrouped, decInGrouped, decOutGrouped = {}, {}, {}      --  key:len, value:{seq1,seq2...seqn} where all of seq1 ... seqn are of length = len
    for i=1, n do

        encSeq, decInSeq, decOutSeq = unpack(data[i]); seqLen = (#encSeq)[1]
        if encGrouped[seqLen] ~= nil and #(encGrouped[seqLen]) == bs-1 then

            table.insert(encGrouped[seqLen], encSeq); table.insert(decInGrouped[seqLen], decInSeq); table.insert(decOutGrouped[seqLen], decOutSeq)

            table.insert(batch, {cat(encGrouped[seqLen], 2):t():contiguous(), cat(decInGrouped[seqLen], 2):t():contiguous(), cat(decOutGrouped[seqLen], 2):t():contiguous()} )
            encGrouped[seqLen] = nil; decInGrouped[seqLen] = nil; decOutGrouped[seqLen] = nil
        else
            encGrouped[seqLen] = encGrouped[seqLen] or {}; decInGrouped[seqLen] = decInGrouped[seqLen] or {}; decOutGrouped[seqLen] = decOutGrouped[seqLen] or {}
            table.insert(encGrouped[seqLen], encSeq); table.insert(decInGrouped[seqLen], decInSeq); table.insert(decOutGrouped[seqLen], decOutSeq)
        end
    end
    
    -- include left overs (last batches which may have sequences lesser than batch size)
    for i, grp in ipairs(encGrouped) do 
        table.insert(batch, {cat(encGrouped[i], 2):t():contiguous(), cat(decInGrouped[i], 2):t():contiguous(), cat(decOutGrouped[i], 2):t():contiguous()} )
    end
end
        

function Synthetic:next_batch(split_index)     -- pass 1 for train, 2 for test
    if split_index == 1 and self.train_index == self.ntrain then
        self.train_index = 0
        return self:next_batch(split_index)
    elseif split_index == 1 then
        self.train_index = self.train_index + 1
        return unpack(self.train_batch[self.train_index])
    elseif split_index == 2 and self.val_index == self.nval then
        self.val_index = 0
        return self:next_batch(split_index)
    elseif split_index == 2 then        
        self.val_index = self.val_index + 1
        return unpack(self.val_batch[self.val_index])
    elseif split_index == 3 and self.test_index == self.ntest then
        self.test_index = 0
        return self:next_batch(split_index)
    elseif split_index == 3 then        
        self.test_index = self.test_index + 1
        return unpack(self.test_batch[self.test_index])
    end
end

return Synthetic
