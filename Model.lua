require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'stn_L1'
require 'stn_L2'

local Model = {}

local conv = cudnn.SpatialConvolution
local batchnorm = cudnn.SpatialBatchNormalization
local relu = cudnn.ReLU
local upsample = nn.SpatialUpSamplingNearest

--[[
local function convBlock(numIn, numOut)
    return nn.Sequential()
        :add(batchnorm(numIn))
        :add(relu(true))
        :add(conv(numIn,numOut/2,1,1))
        :add(batchnorm(numOut/2))
        :add(relu(true))
        :add(conv(numOut/2,numOut/2,3,3,1,1,1,1))
        :add(batchnorm(numOut/2))
        :add(relu(true))
        :add(conv(numOut/2,numOut,1,1))
end
--]]

local function convBlock(inChannels, outChannels)
    local convnet = nn.Sequential()
    convnet:add(cudnn.SpatialBatchNormalization(inChannels))
    convnet:add(cudnn.ReLU(true))
    convnet:add(cudnn.SpatialConvolution(inChannels, outChannels, 3, 3, 1, 1, 1, 1))
    convnet:add(cudnn.SpatialBatchNormalization(outChannels))
    convnet:add(cudnn.ReLU(true))
    convnet:add(cudnn.SpatialConvolution(outChannels, outChannels, 1, 1))
    return convnet
end


-- Skip layer
local function skipLayer(numIn, numOut)
    if numIn == numOut then
        return nn.Identity()
    else
        return nn.Sequential()
            :add(conv(numIn,numOut,1,1))
    end
end


-- Residual block 
function Residual(numIn, numOut)
    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(convBlock(numIn,numOut))
            :add(skipLayer(numIn,numOut)))
        :add(nn.CAddTable(true))
end


local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local l = cudnn.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return cudnn.ReLU(true)(cudnn.SpatialBatchNormalization(numOut)(l))
end

--[[
local function hourglass(n, f, inp)
    -- Upper branch
    local up1 = inp
    up1 = Residual(f,f)(up1)

    -- Lower branch
    local low1 = cudnn.SpatialMaxPooling(2,2,2,2)(inp)
    
    low1 = Residual(f,f)(low1)
    local low2
    if n > 1 then 
        low2 = hourglass(n-1,f,low1)
    else
        low2 = low1
        low2 = Residual(f,f)(low2)
    end

    local low3 = low2
    low3 = Residual(f,f)(low3)
    local up2 = nn.SpatialUpSamplingNearest(2)(low3)

    -- Bring two branches together
    return nn.CAddTable()({up1,up2})
end
--]]
local function hourglass(n, f)
        local model = nn.Sequential()

        local branch = nn.ConcatTable()
        local b1 = nn.Sequential()
        local b2 = nn.Sequential()

        b1:add(Residual(f,f))
        b2:add(cudnn.SpatialMaxPooling(2,2,2,2))

        b2:add(Residual(f,f))

        if n>1 then
                b2:add(hourglass(n-1,f))
        else
                b2:add(Residual(f,f))
        end

        b2:add(upsample(2))

        branch:add(b1):add(b2)
        model:add(branch)

        return model:add(nn.CAddTable())
end


function Model.createLandmark(opts, nfeatures)

    local inp = nn.Identity()()

    -- Initial processing of the image
    local cnv1_ = cudnn.SpatialConvolution(nfeatures,64,5,5,1,1,2,2)(inp)           -- 64
    local cnv1  = cudnn.ReLU(true)(cudnn.SpatialBatchNormalization(64)(cnv1_))
    local r1    = Residual(64,128)(cnv1)

    local out = {}
    local inter = r1

    for i = 1, opts.nStack do
        local hg = hourglass(4, opts.nFeats, inter)

        -- Residual layers at output resolution
        local ll = hg
        ll = Residual(opts.nFeats, opts.nFeats)(ll) 

        -- Linear layer to produce first set of predictions
        ll = lin(opts.nFeats, opts.nFeats,ll)

        -- Predicted heatmaps
        local tmpOut = cudnn.SpatialConvolution(opts.nFeats, opts.nOutChannels,1,1,1,1,0,0)(ll)
        table.insert(out,tmpOut)

        -- Add predictions back
        if i < opt.nStack then
            local ll_ = cudnn.SpatialConvolution(opts.nFeats,opts.nFeats,1,1,1,1,0,0)(ll)
            local tmpOut_ = cudnn.SpatialConvolution(opts.nOutChannels, opts.nFeats,1,1,1,1,0,0)(tmpOut)
            inter = nn.CAddTable()({inter, ll_, tmpOut_})
        end
    end

    -- Final model
    local model = nn.gModule({inp}, out)

    return model
end


function Model.createNetD()
    model_D = nn.Sequential()
    model_D:add(cudnn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2))
    model_D:add(cudnn.SpatialMaxPooling(2,2))
    model_D:add(cudnn.ReLU(true))
    model_D:add(nn.SpatialDropout(0.2))  
    model_D:add(cudnn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2))
    model_D:add(cudnn.SpatialMaxPooling(2,2))
    model_D:add(cudnn.ReLU(true))
    model_D:add(nn.SpatialDropout(0.2))
    model_D:add(cudnn.SpatialConvolution(64, 128, 5, 5, 1, 1, 2, 2))
    model_D:add(cudnn.SpatialMaxPooling(2,2))
    model_D:add(cudnn.ReLU(true))
    model_D:add(nn.SpatialDropout(0.2))
    model_D:add(cudnn.SpatialConvolution(128, 96, 5, 5, 1, 1, 2, 2))
    model_D:add(cudnn.ReLU(true))
    model_D:add(cudnn.SpatialMaxPooling(2,2))
    model_D:add(nn.SpatialDropout(0.2))
    model_D:add(nn.Reshape(8*8*96))
    model_D:add(nn.Linear(8*8*96, 1024))
    model_D:add(cudnn.ReLU(true))
    model_D:add(nn.Dropout())
    model_D:add(nn.Linear(1024,1))
    model_D:add(nn.Sigmoid())

    return model_D
end


function Model.createNetG(opts)

    local ngf = 32
    local input = nn.Identity()()

    local e1 = input - cudnn.SpatialConvolution(3, ngf, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*1)  - nn.LeakyReLU(0.2, true)   -- 8
    local e2 = e1  - cudnn.SpatialConvolution(ngf, ngf*4, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - nn.LeakyReLU(0.2, true)  -- 4
    local e3 = e2  - cudnn.SpatialConvolution(ngf*4, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16)  - nn.LeakyReLU(0.2, true)-- 2
    local e4 = e3  - cudnn.SpatialConvolution(ngf*16, ngf*64, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*64) - nn.LeakyReLU(0.2, true)  -- 1

    local d1 = e4 - cudnn.SpatialFullConvolution(ngf*64, ngf*32, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*32) - cudnn.ReLU(true)  -- 2
    local d2 = {d1, e3} - nn.JoinTable(2, 4)
    local d3 = d2 - cudnn.SpatialFullConvolution(ngf*48, ngf*24, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*24) - cudnn.ReLU(true)  -- 4
    local d4 = {d3, e2} - nn.JoinTable(2, 4)
    local d5 = d4 - cudnn.SpatialFullConvolution(ngf*28, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16) - cudnn.ReLU(true)  -- 8
    local d6 = {d5, e1} - nn.JoinTable(2, 4)
    local d7 = d6 - cudnn.SpatialFullConvolution(ngf*17, ngf*8, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*8) - cudnn.ReLU(true)  -- 16
    local d7_b1 = d7 - cudnn.SpatialFullConvolution(ngf*8, ngf*4, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - cudnn.ReLU(true)  -- 32
    local d7_b2 = input - nn.SpatialUpSamplingNearest(2)

    local d8 = {d7_b2, d7_b1} - nn.JoinTable(2, 4)

    local d9 = d8 - stn_L1(ngf*4+3) - cudnn.SpatialConvolution(ngf*4+3, ngf*4, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - cudnn.ReLU(true) -- 32
    local d10 = d9 - nn.SpatialUpSamplingNearest(2)

	if opts.num_stn == 2 then
		d10 = d10 - stn_L2(ngf*4)
	end
	
	local d11 = d10 - cudnn.SpatialConvolution(ngf*4, ngf* 2, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(ngf*2) - cudnn.ReLU(true) -- 64

    num_features = 64
    -- Initial processing of the image
	local conv1 = conv(ngf*2, ngf, 3, 3, 1, 1,1,1)(d11)
    local r1 = Residual(ngf, num_features)(conv1)

    local out = {}
    local inter = r1

    for i = 1, opts.nStack do
        
        local hg = hourglass(3, num_features)(inter)

        -- Residual layers at output resolution
        local ll = Residual(num_features, num_features)(hg) 

        -- Linear layer to produce first set of predictions
        local LL = lin(num_features, num_features, ll)

        -- Predicted heatmaps
        local tmpOut = cudnn.SpatialConvolution(num_features, opts.nOutChannels, 1,1,1,1,0,0)(LL)
        table.insert(out,tmpOut)

        -- Add predictions back
        if i < opt.nStack then
            local ll_ = cudnn.SpatialConvolution(num_features, num_features,1,1,1,1,0,0)(LL)
            local tmpOut_ = cudnn.SpatialConvolution(opts.nOutChannels, num_features,1,1,1,1,0,0)(tmpOut)
            inter = nn.CAddTable()({inter, ll_, tmpOut_})
        end
    end

    local d11_ = {d11, out[opts.nStack]} - nn.JoinTable(2, 4)
    local d12 = d11_ - cudnn.SpatialConvolution(ngf*2 + opts.nOutChannels, ngf* 2, 3, 3, 1, 1, 1, 1) - Residual(ngf*2, ngf*2) - cudnn.ReLU(true)
    local d13 = d12 - nn.SpatialUpSamplingNearest(2) - cudnn.SpatialConvolution(ngf*2, ngf*1, 3, 3, 1, 1, 1, 1) - Residual(ngf, ngf) - Residual(ngf, ngf) - cudnn.ReLU(true)
    local d14 = d13 - cudnn.SpatialConvolution(ngf, ngf/2, 3, 3, 1, 1, 1, 1) - Residual(ngf/2, 3)

    table.insert(out, d14)

    -- Final model
    local model = nn.gModule({input}, out)

    return model
end


function Model.createNetG_v2(opts)

    local ngf = 32
    local input = nn.Identity()()

    local e1 = input - cudnn.SpatialConvolution(3, ngf, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*1)  - nn.LeakyReLU(0.2, true)   -- 8
    local e2 = e1  - cudnn.SpatialConvolution(ngf, ngf*4, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - nn.LeakyReLU(0.2, true)  -- 4
    local e3 = e2  - cudnn.SpatialConvolution(ngf*4, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16)  - nn.LeakyReLU(0.2, true)-- 2
    local e4 = e3  - cudnn.SpatialConvolution(ngf*16, ngf*64, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*64) - nn.LeakyReLU(0.2, true)  -- 1

    local d1 = e4 - cudnn.SpatialFullConvolution(ngf*64, ngf*32, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*32) - cudnn.ReLU(true)  -- 2
    local d2 = {d1, e3} - nn.JoinTable(2, 4)
    local d3 = d2 - cudnn.SpatialFullConvolution(ngf*48, ngf*24, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*24) - cudnn.ReLU(true)  -- 4
    local d4 = {d3, e2} - nn.JoinTable(2, 4)
    local d5 = d4 - cudnn.SpatialFullConvolution(ngf*28, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16) - cudnn.ReLU(true)  -- 8
    local d6 = {d5, e1} - nn.JoinTable(2, 4)
    local d7 = d6 - cudnn.SpatialFullConvolution(ngf*17, ngf*8, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*8) - cudnn.ReLU(true)  -- 16
    
    local d7_16 = {input, d7} - nn.JoinTable(2, 4)
    local d7_32 = d7_16 - cudnn.SpatialFullConvolution(ngf*8+3, ngf*4, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - cudnn.ReLU(true)  -- 32
    local d7_b1 = d7_32 - cudnn.SpatialFullConvolution(ngf*4, ngf*2, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*2) - cudnn.ReLU(true) -- 64

    local d7_b2 = input - nn.SpatialUpSamplingNearest(4)
    local d8 = {d7_b2, d7_b1} - nn.JoinTable(2, 4)

    local d9 = d8 - stn_L2(ngf*2+3) - cudnn.SpatialConvolution(ngf*2+3, ngf*3, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(ngf*3) - cudnn.ReLU(true) -- 64
    
    num_features = 64
    -- Initial processing of the image
	local conv1 = conv(ngf*3, ngf, 3, 3, 1, 1,1,1)(d9)
    local r1 = Residual(ngf, num_features)(conv1)

    local out = {}
    local inter = r1

    for i = 1, opts.nStack do
        
        local hg = hourglass(4, num_features)(inter)

        -- Residual layers at output resolution
        local ll = Residual(num_features, num_features)(hg) 

        -- Linear layer to produce first set of predictions
        local LL = lin(num_features, num_features, ll)

        -- Predicted heatmaps
        local tmpOut = cudnn.SpatialConvolution(num_features, opts.nOutChannels, 1,1,1,1,0,0)(LL)
        table.insert(out,tmpOut)

        -- Add predictions back
        if i < opt.nStack then
            local ll_ = cudnn.SpatialConvolution(num_features, num_features,1,1,1,1,0,0)(LL)
            local tmpOut_ = cudnn.SpatialConvolution(opts.nOutChannels, num_features,1,1,1,1,0,0)(tmpOut)
            inter = nn.CAddTable()({inter, ll_, tmpOut_})
        end
    end

    local d11_ = {d9, out[opts.nStack]} - nn.JoinTable(2, 4)
    local d12 = d11_ - cudnn.SpatialConvolution(ngf*3 + opts.nOutChannels, ngf* 4, 3, 3, 1, 1, 1, 1) - Residual(ngf*4, ngf*2) - Residual(ngf*2, ngf*2) - cudnn.ReLU(true)
    local d13 = d12 - nn.SpatialUpSamplingNearest(2) - cudnn.SpatialConvolution(ngf*2, ngf*2, 3, 3, 1, 1, 1, 1) - Residual(ngf*2, ngf) - Residual(ngf, ngf) - cudnn.ReLU(true)
    local d14 = d13 - cudnn.SpatialConvolution(ngf, ngf/2, 3, 3, 1, 1, 1, 1) - Residual(ngf/2, ngf/2) - cudnn.ReLU(true) - cudnn.SpatialConvolution(ngf/2, 3, 5, 5, 1, 1, 2, 2)

    table.insert(out, d14)

    -- Final model
    local model = nn.gModule({input}, out)

    return model
end


function Model.createNetG_noLM(opts)

    local ngf = 32
    local input = nn.Identity()()

    local e1 = input - cudnn.SpatialConvolution(3, ngf, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*1)  - nn.LeakyReLU(0.2, true)   -- 8
    local e2 = e1  - cudnn.SpatialConvolution(ngf, ngf*4, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - nn.LeakyReLU(0.2, true)  -- 4
    local e3 = e2  - cudnn.SpatialConvolution(ngf*4, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16)  - nn.LeakyReLU(0.2, true)-- 2
    local e4 = e3  - cudnn.SpatialConvolution(ngf*16, ngf*64, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*64) - nn.LeakyReLU(0.2, true)  -- 1

    local d1 = e4 - cudnn.SpatialFullConvolution(ngf*64, ngf*32, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*32) - cudnn.ReLU(true)  -- 2
    local d2 = {d1, e3} - nn.JoinTable(2, 4)
    local d3 = d2 - cudnn.SpatialFullConvolution(ngf*48, ngf*24, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*24) - cudnn.ReLU(true)  -- 4
    local d4 = {d3, e2} - nn.JoinTable(2, 4)
    local d5 = d4 - cudnn.SpatialFullConvolution(ngf*28, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16) - cudnn.ReLU(true)  -- 8
    local d6 = {d5, e1} - nn.JoinTable(2, 4)
    local d7 = d6 - cudnn.SpatialFullConvolution(ngf*17, ngf*8, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*8) - cudnn.ReLU(true)  -- 16
    
    local d7_16 = {input, d7} - nn.JoinTable(2, 4)
    local d7_32 = d7_16 - cudnn.SpatialFullConvolution(ngf*8+3, ngf*4, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - cudnn.ReLU(true)  -- 32
    local d7_b1 = d7_32 - cudnn.SpatialFullConvolution(ngf*4, ngf*2, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*2) - cudnn.ReLU(true) -- 64

    local d7_b2 = input - nn.SpatialUpSamplingNearest(4)
    local d8 = {d7_b2, d7_b1} - nn.JoinTable(2, 4)

    local d9 = d8 - stn_L2(ngf*2+3) - cudnn.SpatialConvolution(ngf*2+3, ngf*3, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(ngf*3) - cudnn.ReLU(true) -- 64

    local d12 = d9 - cudnn.SpatialConvolution(ngf*3, ngf* 4, 3, 3, 1, 1, 1, 1) - Residual(ngf*4, ngf*2) - Residual(ngf*2, ngf*2) - cudnn.ReLU(true)
    local d13 = d12 - nn.SpatialUpSamplingNearest(2) - cudnn.SpatialConvolution(ngf*2, ngf*2, 3, 3, 1, 1, 1, 1) - Residual(ngf*2, ngf) - Residual(ngf, ngf) - cudnn.ReLU(true)
    local d14 = d13 - cudnn.SpatialConvolution(ngf, ngf/2, 3, 3, 1, 1, 1, 1) - Residual(ngf/2, ngf/2) - cudnn.ReLU(true) - cudnn.SpatialConvolution(ngf/2, 3, 5, 5, 1, 1, 2, 2)


    -- Final model
    local model = nn.gModule({input}, {d14})

    return model
end


function Model.createNetG_skip(opts)

    local ngf = 32
    local input = nn.Identity()()

    local e1 = input - cudnn.SpatialConvolution(3, ngf, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*1)  - nn.LeakyReLU(0.2, true)   -- 8
    local e2 = e1  - cudnn.SpatialConvolution(ngf, ngf*4, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - nn.LeakyReLU(0.2, true)  -- 4
    local e3 = e2  - cudnn.SpatialConvolution(ngf*4, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16)  - nn.LeakyReLU(0.2, true)-- 2
    local e4 = e3  - cudnn.SpatialConvolution(ngf*16, ngf*64, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*64) - nn.LeakyReLU(0.2, true)  -- 1

    local d1 = e4 - cudnn.SpatialFullConvolution(ngf*64, ngf*32, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*32) - cudnn.ReLU(true)  -- 2
    local d2 = {d1, e3} - nn.JoinTable(2, 4)
    local d3 = d2 - cudnn.SpatialFullConvolution(ngf*48, ngf*24, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*24) - cudnn.ReLU(true)  -- 4
    local d4 = {d3, e2} - nn.JoinTable(2, 4)
    local d5 = d4 - cudnn.SpatialFullConvolution(ngf*28, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16) - cudnn.ReLU(true)  -- 8
    local d6 = {d5, e1} - nn.JoinTable(2, 4)
    local d7 = d6 - cudnn.SpatialFullConvolution(ngf*17, ngf*8, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*8) - cudnn.ReLU(true)  -- 16
    
    local d7_16 = {input, d7} - nn.JoinTable(2, 4)
    local d7_32 = d7_16 - cudnn.SpatialFullConvolution(ngf*8+3, ngf*4, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - cudnn.ReLU(true)  -- 32
    local d7_b1 = d7_32 - cudnn.SpatialFullConvolution(ngf*4, ngf*2, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*2) - cudnn.ReLU(true) -- 64

    local d7_b2 = input - nn.SpatialUpSamplingNearest(4)
    local d8 = {d7_b2, d7_b1} - nn.JoinTable(2, 4)

    local d9 = d8 - stn_L2(ngf*2+3) - cudnn.SpatialConvolution(ngf*2+3, ngf*2, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(ngf*2) - cudnn.ReLU(true) -- 64
    
    num_features = 64
    -- Initial processing of the image
	  local conv1 = conv(ngf*2, ngf*2, 3, 3, 1, 1,1,1)(d9)
    local r1 = Residual(ngf*2, num_features)(conv1)

    local out = {}
    local inter = r1

    for i = 1, opts.nStack do
        
        local hg = hourglass(4, num_features)(inter)

        -- Residual layers at output resolution
        local ll = Residual(num_features, num_features)(hg) 

        -- Linear layer to produce first set of predictions
        local LL = lin(num_features, num_features, ll)

        -- Predicted heatmaps
        local tmpOut = cudnn.SpatialConvolution(num_features, opts.nOutChannels, 1,1,1,1,0,0)(LL)
        table.insert(out,tmpOut)

        -- Add predictions back
        if i < opt.nStack then
            local ll_ = cudnn.SpatialConvolution(num_features, num_features,1,1,1,1,0,0)(LL)
            local tmpOut_ = cudnn.SpatialConvolution(opts.nOutChannels, num_features,1,1,1,1,0,0)(tmpOut)
            inter = nn.CAddTable()({inter, ll_, tmpOut_})
        end
    end

    local d11_ = {d9, out[opts.nStack]} - nn.JoinTable(2, 4)
    local d12 = d11_ - cudnn.SpatialConvolution(ngf*2 + opts.nOutChannels, ngf* 2, 3, 3, 1, 1, 1, 1) - Residual(ngf*2, ngf*2) -cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) 
    local d12_b1 = d12 - cudnn.SpatialConvolution(ngf*2, ngf, 3, 3, 1, 1, 1, 1)- Residual(ngf, ngf) - cudnn.ReLU(true)
    local d12_b2 = d9 - nn.SpatialUpSamplingNearest(2) - cudnn.SpatialConvolution(ngf*2, ngf*2, 3, 3, 1, 1, 1, 1) - Residual(ngf*2, ngf*2) - cudnn.ReLU(true)
    
    local d12_ = {d12_b2, d12_b1} - nn.JoinTable(2, 4)
    
    local d13 = d12_ - cudnn.SpatialConvolution(ngf*3, ngf*2, 3, 3, 1, 1, 1, 1) - Residual(ngf*2, ngf) - Residual(ngf, ngf) - cudnn.ReLU(true)
    local d14 = d13 - cudnn.SpatialConvolution(ngf, ngf/2, 3, 3, 1, 1, 1, 1) - Residual(ngf/2, ngf/2) - cudnn.ReLU(true) - cudnn.SpatialConvolution(ngf/2, 3, 5, 5, 1, 1, 2, 2)

    table.insert(out, d14)

    -- Final model
    local model = nn.gModule({input}, out)


    return model
end



function Model.createNetG_noAE(opts)

    local ngf = 32
    local input = nn.Identity()()
    
    local d7_32 = input - cudnn.SpatialFullConvolution(3, ngf*4, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - cudnn.ReLU(true)  -- 32
    local d7_b1 = d7_32 - cudnn.SpatialFullConvolution(ngf*4, ngf*2, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*2) - cudnn.ReLU(true) -- 64

    local d7_b2 = input - nn.SpatialUpSamplingNearest(4)
    local d8 = {d7_b2, d7_b1} - nn.JoinTable(2, 4)

    local d9 = d8 - stn_L2(ngf*2+3) - cudnn.SpatialConvolution(ngf*2+3, ngf*2, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(ngf*2) - cudnn.ReLU(true) -- 64
    
    num_features = 64
    -- Initial processing of the image
	  local conv1 = conv(ngf*2, ngf*2, 3, 3, 1, 1,1,1)(d9)
    local r1 = Residual(ngf*2, num_features)(conv1)

    local out = {}
    local inter = r1

    for i = 1, opts.nStack do
        
        local hg = hourglass(4, num_features)(inter)

        -- Residual layers at output resolution
        local ll = Residual(num_features, num_features)(hg) 

        -- Linear layer to produce first set of predictions
        local LL = lin(num_features, num_features, ll)

        -- Predicted heatmaps
        local tmpOut = cudnn.SpatialConvolution(num_features, opts.nOutChannels, 1,1,1,1,0,0)(LL)
        table.insert(out,tmpOut)

        -- Add predictions back
        if i < opt.nStack then
            local ll_ = cudnn.SpatialConvolution(num_features, num_features,1,1,1,1,0,0)(LL)
            local tmpOut_ = cudnn.SpatialConvolution(opts.nOutChannels, num_features,1,1,1,1,0,0)(tmpOut)
            inter = nn.CAddTable()({inter, ll_, tmpOut_})
        end
    end

    local d11_ = {d9, out[opts.nStack]} - nn.JoinTable(2, 4)
    local d12 = d11_ - cudnn.SpatialConvolution(ngf*2 + opts.nOutChannels, ngf* 2, 3, 3, 1, 1, 1, 1) - Residual(ngf*2, ngf*2) -cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) 
    local d12_b1 = d12 - cudnn.SpatialConvolution(ngf*2, ngf, 3, 3, 1, 1, 1, 1)- Residual(ngf, ngf) - cudnn.ReLU(true)
    local d12_b2 = d9 - nn.SpatialUpSamplingNearest(2) - cudnn.SpatialConvolution(ngf*2, ngf*2, 3, 3, 1, 1, 1, 1) - Residual(ngf*2, ngf*2) - cudnn.ReLU(true)
    
    local d12_ = {d12_b2, d12_b1} - nn.JoinTable(2, 4)
    
    local d13 = d12_ - cudnn.SpatialConvolution(ngf*3, ngf*2, 3, 3, 1, 1, 1, 1) - Residual(ngf*2, ngf) - Residual(ngf, ngf) - cudnn.ReLU(true)
    local d14 = d13 - cudnn.SpatialConvolution(ngf, ngf/2, 3, 3, 1, 1, 1, 1) - Residual(ngf/2, ngf/2) - cudnn.ReLU(true) - cudnn.SpatialConvolution(ngf/2, 3, 5, 5, 1, 1, 2, 2)

    table.insert(out, d14)

    -- Final model
    local model = nn.gModule({input}, out)


    return model
end



function Model.createNetG_noskip(opts)

    local ngf = 32
    local input = nn.Identity()()

    local e1 = input - cudnn.SpatialConvolution(3, ngf, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*1)  - nn.LeakyReLU(0.2, true)   -- 8
    local e2 = e1  - cudnn.SpatialConvolution(ngf, ngf*4, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - nn.LeakyReLU(0.2, true)  -- 4
    local e3 = e2  - cudnn.SpatialConvolution(ngf*4, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16)  - nn.LeakyReLU(0.2, true)-- 2
    local e4 = e3  - cudnn.SpatialConvolution(ngf*16, ngf*64, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*64) - nn.LeakyReLU(0.2, true)  -- 1

    local d1 = e4 - cudnn.SpatialFullConvolution(ngf*64, ngf*32, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*32) - cudnn.ReLU(true)  -- 2
    local d2 = {d1, e3} - nn.JoinTable(2, 4)
    local d3 = d2 - cudnn.SpatialFullConvolution(ngf*48, ngf*24, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*24) - cudnn.ReLU(true)  -- 4
    local d4 = {d3, e2} - nn.JoinTable(2, 4)
    local d5 = d4 - cudnn.SpatialFullConvolution(ngf*28, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16) - cudnn.ReLU(true)  -- 8
    local d6 = {d5, e1} - nn.JoinTable(2, 4)
    local d7 = d6 - cudnn.SpatialFullConvolution(ngf*17, ngf*8, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*8) - cudnn.ReLU(true)  -- 16
    
    local d7_16 = {input, d7} - nn.JoinTable(2, 4)
    local d7_32 = d7_16 - cudnn.SpatialFullConvolution(ngf*8+3, ngf*4, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - cudnn.ReLU(true)  -- 32
    local d7_b1 = d7_32 - cudnn.SpatialFullConvolution(ngf*4, ngf*2, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*2) - cudnn.ReLU(true) -- 64

    local d7_b2 = input - nn.SpatialUpSamplingNearest(4)
    local d8 = {d7_b2, d7_b1} - nn.JoinTable(2, 4)

    local d9 = d8 - stn_L2(ngf*2+3) - cudnn.SpatialConvolution(ngf*2+3, ngf*2, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(ngf*2) - cudnn.ReLU(true) -- 64
    
    num_features = 64
    -- Initial processing of the image
	  local conv1 = conv(ngf*2, ngf*2, 3, 3, 1, 1,1,1)(d9)
    local r1 = Residual(ngf*2, num_features)(conv1)

    local out = {}
    local inter = r1

    for i = 1, opts.nStack do
        
        local hg = hourglass(4, num_features)(inter)

        -- Residual layers at output resolution
        local ll = Residual(num_features, num_features)(hg) 

        -- Linear layer to produce first set of predictions
        local LL = lin(num_features, num_features, ll)

        -- Predicted heatmaps
        local tmpOut = cudnn.SpatialConvolution(num_features, opts.nOutChannels, 1,1,1,1,0,0)(LL)
        table.insert(out,tmpOut)

        -- Add predictions back
        if i < opt.nStack then
            local ll_ = cudnn.SpatialConvolution(num_features, num_features,1,1,1,1,0,0)(LL)
            local tmpOut_ = cudnn.SpatialConvolution(opts.nOutChannels, num_features,1,1,1,1,0,0)(tmpOut)
            inter = nn.CAddTable()({inter, ll_, tmpOut_})
        end
    end

    local d11_ = {d9, out[opts.nStack]} - nn.JoinTable(2, 4)
    local d12 = d11_ - cudnn.SpatialConvolution(ngf*2 + opts.nOutChannels, ngf* 2, 3, 3, 1, 1, 1, 1) - Residual(ngf*2, ngf*2) -cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) 
    local d12_b1 = d12 - cudnn.SpatialConvolution(ngf*2, ngf, 3, 3, 1, 1, 1, 1)- Residual(ngf, ngf) - cudnn.ReLU(true)
--    local d12_b2 = d9 - nn.SpatialUpSamplingNearest(2) - cudnn.SpatialConvolution(ngf*2, ngf*2, 3, 3, 1, 1, 1, 1) - Residual(ngf*2, ngf*2) - cudnn.ReLU(true)
    
--    local d12_ = {d12_b2, d12_b1} - nn.JoinTable(2, 4)
    
    local d13 = d12_b1 - cudnn.SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1) - Residual(ngf, ngf) - Residual(ngf, ngf) - cudnn.ReLU(true)
    local d14 = d13 - cudnn.SpatialConvolution(ngf, ngf/2, 3, 3, 1, 1, 1, 1) - Residual(ngf/2, ngf/2) - cudnn.ReLU(true) - cudnn.SpatialConvolution(ngf/2, 3, 5, 5, 1, 1, 2, 2)

    table.insert(out, d14)

    -- Final model
    local model = nn.gModule({input}, out)


    return model
end


function Model.createNetG_skip_v1(opts)

    local ngf = 32
    local input = nn.Identity()()

    local e1 = input - cudnn.SpatialConvolution(3, ngf, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*1)  - nn.LeakyReLU(0.2, true)   -- 8
    local e2 = e1  - cudnn.SpatialConvolution(ngf, ngf*4, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - nn.LeakyReLU(0.2, true)  -- 4
    local e3 = e2  - cudnn.SpatialConvolution(ngf*4, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16)  - nn.LeakyReLU(0.2, true)-- 2
    local e4 = e3  - cudnn.SpatialConvolution(ngf*16, ngf*64, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*64) - nn.LeakyReLU(0.2, true)  -- 1

    local d1 = e4 - cudnn.SpatialFullConvolution(ngf*64, ngf*32, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*32) - cudnn.ReLU(true)  -- 2
    local d2 = {d1, e3} - nn.JoinTable(2, 4)
    local d3 = d2 - cudnn.SpatialFullConvolution(ngf*48, ngf*24, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*24) - cudnn.ReLU(true)  -- 4
    local d4 = {d3, e2} - nn.JoinTable(2, 4)
    local d5 = d4 - cudnn.SpatialFullConvolution(ngf*28, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16) - cudnn.ReLU(true)  -- 8
    local d6 = {d5, e1} - nn.JoinTable(2, 4)
    local d7 = d6 - cudnn.SpatialFullConvolution(ngf*17, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16) - cudnn.ReLU(true)  -- 16
    
    --local d7_16 = {input, d7} - nn.JoinTable(2, 4)
    local d7_32 = d7 - cudnn.SpatialFullConvolution(ngf*16, ngf*8, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*8) - cudnn.ReLU(true)  -- 32
    local d7_b1 = d7_32 - cudnn.SpatialFullConvolution(ngf*8, ngf*8, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*8) - cudnn.ReLU(true) -- 64

    --local d7_b2 = input - nn.SpatialUpSamplingNearest(4)   - using this the final results are not good
    --local d8 = {d7_b2, d7_b1} - nn.JoinTable(2, 4)
    
    local d9 = d7_b1 - stn_L2(ngf*8) - cudnn.SpatialConvolution(ngf*8, ngf*4, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - cudnn.ReLU(true) -- 64
    
    num_features = 64
    -- Initial processing of the image
	  local conv1 = conv(ngf*4, ngf*2, 3, 3, 1, 1,1,1)(d9)
    local r1 = Residual(ngf*2, num_features)(conv1)

    local out = {}
    local inter = r1

    for i = 1, opts.nStack do
        
        local hg = hourglass(4, num_features)(inter)

        -- Residual layers at output resolution
        local ll = Residual(num_features, num_features)(hg) 

        -- Linear layer to produce first set of predictions
        local LL = lin(num_features, num_features, ll)

        -- Predicted heatmaps
        local tmpOut = cudnn.SpatialConvolution(num_features, opts.nOutChannels, 1,1,1,1,0,0)(LL)
        table.insert(out,tmpOut)

        -- Add predictions back
        if i < opt.nStack then
            local ll_ = cudnn.SpatialConvolution(num_features, num_features,1,1,1,1,0,0)(LL)
            local tmpOut_ = cudnn.SpatialConvolution(opts.nOutChannels, num_features,1,1,1,1,0,0)(tmpOut)
            inter = nn.CAddTable()({inter, ll_, tmpOut_})
        end
    end

    local d11_ = {d9, out[opts.nStack]} - nn.JoinTable(2, 4)
    local d12 = d11_ - cudnn.SpatialConvolution(ngf*4 + opts.nOutChannels, ngf* 4, 3, 3, 1, 1, 1, 1) - Residual(ngf*4, ngf*2) -cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) 
    local d12_b1 = d12 - cudnn.SpatialConvolution(ngf*2, ngf, 3, 3, 1, 1, 1, 1)- Residual(ngf, ngf) - cudnn.ReLU(true)
    
    local d12_b2 = d9 - nn.SpatialUpSamplingNearest(2) - cudnn.SpatialConvolution(ngf*4, ngf*2, 3, 3, 1, 1, 1, 1) - Residual(ngf*2, ngf*2) - cudnn.ReLU(true)
    
    local d12_ = {d12_b2, d12_b1} - nn.JoinTable(2, 4)
    
    local d13 = d12_ - cudnn.SpatialConvolution(ngf*3, ngf*2, 3, 3, 1, 1, 1, 1) - Residual(ngf*2, ngf) - Residual(ngf, ngf) - cudnn.ReLU(true)
    local d14 = d13 - cudnn.SpatialConvolution(ngf, ngf/2, 3, 3, 1, 1, 1, 1) - Residual(ngf/2, ngf/2) - cudnn.ReLU(true) - cudnn.SpatialConvolution(ngf/2, 3, 5, 5, 1, 1, 2, 2)

    table.insert(out, d14)

    -- Final model
    local model = nn.gModule({input}, out)


    return model
end


function Model.createNetG_skip_v2(opts)

    local ngf = 32
    local input = nn.Identity()()

    local e1 = input - cudnn.SpatialConvolution(3, ngf, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*1)  - nn.LeakyReLU(0.2, true)   -- 8
    local e2 = e1  - cudnn.SpatialConvolution(ngf, ngf*4, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - nn.LeakyReLU(0.2, true)  -- 4
    local e3 = e2  - cudnn.SpatialConvolution(ngf*4, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16)  - nn.LeakyReLU(0.2, true)-- 2
    local e4 = e3  - cudnn.SpatialConvolution(ngf*16, ngf*64, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*64) - nn.LeakyReLU(0.2, true)  -- 1

    local d1 = e4 - cudnn.SpatialFullConvolution(ngf*64, ngf*32, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*32) - cudnn.ReLU(true)  -- 2
    local d2 = {d1, e3} - nn.JoinTable(2, 4)
    local d3 = d2 - cudnn.SpatialFullConvolution(ngf*48, ngf*24, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*24) - cudnn.ReLU(true)  -- 4
    local d4 = {d3, e2} - nn.JoinTable(2, 4)
    local d5 = d4 - cudnn.SpatialFullConvolution(ngf*28, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16) - cudnn.ReLU(true)  -- 8
    local d6 = {d5, e1} - nn.JoinTable(2, 4)
    local d7 = d6 - cudnn.SpatialFullConvolution(ngf*17, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16) - cudnn.ReLU(true)  -- 16
    local d7_ = d7 - cudnn.SpatialFullConvolution(ngf*16, ngf*8, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*8) - cudnn.ReLU(true)  -- 32
    
    local d6_1 = input - cudnn.SpatialFullConvolution(3, ngf*8, 4, 4, 2, 2, 1, 1) - Residual(ngf*8, ngf*8) - cudnn.ReLU(true)  -- 32
    local d7_32 = {d6_1, d7_} - nn.JoinTable(2, 4)
    
    local d7_b1 = d7_32 - cudnn.SpatialFullConvolution(ngf*16, ngf*8, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*8) - cudnn.ReLU(true) -- 64

    --local d7_b2 = input - nn.SpatialUpSamplingNearest(4)   - using this the final results are not good
    --local d8 = {d7_b2, d7_b1} - nn.JoinTable(2, 4)
    
    local d9 = d7_b1 - stn_L2(ngf*8) - cudnn.SpatialConvolution(ngf*8, ngf*4, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - cudnn.ReLU(true) -- 64
    
    num_features = 64
    -- Initial processing of the image
	  local conv1 = conv(ngf*4, ngf*2, 3, 3, 1, 1,1,1)(d9)
    local r1 = Residual(ngf*2, num_features)(conv1)

    local out = {}
    local inter = r1

    for i = 1, opts.nStack do
        
        local hg = hourglass(4, num_features)(inter)

        -- Residual layers at output resolution
        local ll = Residual(num_features, num_features)(hg) 

        -- Linear layer to produce first set of predictions
        local LL = lin(num_features, num_features, ll)

        -- Predicted heatmaps
        local tmpOut = cudnn.SpatialConvolution(num_features, opts.nOutChannels, 1,1,1,1,0,0)(LL)
        table.insert(out,tmpOut)

        -- Add predictions back
        if i < opt.nStack then
            local ll_ = cudnn.SpatialConvolution(num_features, num_features,1,1,1,1,0,0)(LL)
            local tmpOut_ = cudnn.SpatialConvolution(opts.nOutChannels, num_features,1,1,1,1,0,0)(tmpOut)
            inter = nn.CAddTable()({inter, ll_, tmpOut_})
        end
    end

    local d11_ = {d9, out[opts.nStack]} - nn.JoinTable(2, 4)
    local d12 = d11_ - cudnn.SpatialConvolution(ngf*4 + opts.nOutChannels, ngf* 4, 3, 3, 1, 1, 1, 1) - Residual(ngf*4, ngf*2) -cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) 
    local d12_b1 = d12 - cudnn.SpatialConvolution(ngf*2, ngf, 3, 3, 1, 1, 1, 1)- Residual(ngf, ngf) - cudnn.ReLU(true)
    
    local d12_b2 = d9 - nn.SpatialUpSamplingNearest(2) - cudnn.SpatialConvolution(ngf*4, ngf*2, 3, 3, 1, 1, 1, 1) - Residual(ngf*2, ngf*2) - cudnn.ReLU(true)
    
    local d12_ = {d12_b2, d12_b1} - nn.JoinTable(2, 4)
    
    local d13 = d12_ - cudnn.SpatialConvolution(ngf*3, ngf*2, 3, 3, 1, 1, 1, 1) - Residual(ngf*2, ngf) - Residual(ngf, ngf) - cudnn.ReLU(true)
    local d14 = d13 - cudnn.SpatialConvolution(ngf, ngf/2, 3, 3, 1, 1, 1, 1) - Residual(ngf/2, ngf/2) - cudnn.ReLU(true) - cudnn.SpatialConvolution(ngf/2, 3, 5, 5, 1, 1, 2, 2)

    table.insert(out, d14)

    -- Final model
    local model = nn.gModule({input}, out)


    return model
end



function Model.createNetG_skip_16(opts)

    local ngf = 32
    local input = nn.Identity()()

    local e1 = input - cudnn.SpatialConvolution(3, ngf, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*1)  - nn.LeakyReLU(0.2, true)   -- 8
    local e2 = e1  - cudnn.SpatialConvolution(ngf, ngf*4, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - nn.LeakyReLU(0.2, true)  -- 4
    local e3 = e2  - cudnn.SpatialConvolution(ngf*4, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16)  - nn.LeakyReLU(0.2, true)-- 2
    local e4 = e3  - cudnn.SpatialConvolution(ngf*16, ngf*64, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*64) - nn.LeakyReLU(0.2, true)  -- 1

    local d1 = e4 - cudnn.SpatialFullConvolution(ngf*64, ngf*32, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*32) - cudnn.ReLU(true)  -- 2
    local d2 = {d1, e3} - nn.JoinTable(2, 4)
    local d3 = d2 - cudnn.SpatialFullConvolution(ngf*48, ngf*24, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*24) - cudnn.ReLU(true)  -- 4
    local d4 = {d3, e2} - nn.JoinTable(2, 4)
    local d5 = d4 - cudnn.SpatialFullConvolution(ngf*28, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16) - cudnn.ReLU(true)  -- 8
    local d6 = {d5, e1} - nn.JoinTable(2, 4)
    local d7 = d6 - cudnn.SpatialFullConvolution(ngf*17, ngf*8, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*8) - cudnn.ReLU(true)  -- 16
    
    local d7_16 = {input, d7} - nn.JoinTable(2, 4)
    
    num_features = 64
    -- Initial processing of the image
	  local conv1 = conv(ngf*8+3, ngf*2, 3, 3, 1, 1,1,1)(d7_16)
    local r1 = Residual(ngf*2, num_features)(conv1)

    local out = {}
    local inter = r1

    for i = 1, opts.nStack do
        
        local hg = hourglass(4, num_features)(inter)

        -- Residual layers at output resolution
        local ll = Residual(num_features, num_features)(hg) 

        -- Linear layer to produce first set of predictions
        local LL = lin(num_features, num_features, ll)

        -- Predicted heatmaps
        local tmpOut = cudnn.SpatialConvolution(num_features, opts.nOutChannels, 1,1,1,1,0,0)(LL)
        table.insert(out,tmpOut)

        -- Add predictions back
        if i < opt.nStack then
            local ll_ = cudnn.SpatialConvolution(num_features, num_features,1,1,1,1,0,0)(LL)
            local tmpOut_ = cudnn.SpatialConvolution(opts.nOutChannels, num_features,1,1,1,1,0,0)(tmpOut)
            inter = nn.CAddTable()({inter, ll_, tmpOut_})
        end
    end
    
    
    local d8 = {d7_16, out[opts.nStack]} - nn.JoinTable(2, 4)
    
    local d9 = d8 - cudnn.SpatialFullConvolution(ngf*8+3+opts.nOutChannels, ngf*4, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - cudnn.ReLU(true)  -- 32
    
    local d10 = d9 - cudnn.SpatialFullConvolution(ngf*4, ngf*2, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*2) - cudnn.ReLU(true) -- 64

    local d11 = d10 - stn_L2(ngf*2) - cudnn.SpatialConvolution(ngf*2, ngf*2, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(ngf*2) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) -- 128
    
    local d13 = d11 - cudnn.SpatialConvolution(ngf*2, ngf*2, 3, 3, 1, 1, 1, 1) - Residual(ngf*2, ngf) - Residual(ngf, ngf) - cudnn.ReLU(true)
    local d14 = d13 - cudnn.SpatialConvolution(ngf, ngf/2, 3, 3, 1, 1, 1, 1) - Residual(ngf/2, ngf/2) - cudnn.ReLU(true) - cudnn.SpatialConvolution(ngf/2, 3, 5, 5, 1, 1, 2, 2)

    table.insert(out, d14)

    -- Final model
    local model = nn.gModule({input}, out)


    return model
end

function Model.createNetG_skip_32(opts)

    local ngf = 32
    local input = nn.Identity()()

    local e1 = input - cudnn.SpatialConvolution(3, ngf, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*1)  - nn.LeakyReLU(0.2, true)   -- 8
    local e2 = e1  - cudnn.SpatialConvolution(ngf, ngf*4, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - nn.LeakyReLU(0.2, true)  -- 4
    local e3 = e2  - cudnn.SpatialConvolution(ngf*4, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16)  - nn.LeakyReLU(0.2, true)-- 2
    local e4 = e3  - cudnn.SpatialConvolution(ngf*16, ngf*64, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*64) - nn.LeakyReLU(0.2, true)  -- 1

    local d1 = e4 - cudnn.SpatialFullConvolution(ngf*64, ngf*32, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*32) - cudnn.ReLU(true)  -- 2
    local d2 = {d1, e3} - nn.JoinTable(2, 4)
    local d3 = d2 - cudnn.SpatialFullConvolution(ngf*48, ngf*24, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*24) - cudnn.ReLU(true)  -- 4
    local d4 = {d3, e2} - nn.JoinTable(2, 4)
    local d5 = d4 - cudnn.SpatialFullConvolution(ngf*28, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16) - cudnn.ReLU(true)  -- 8
    local d6 = {d5, e1} - nn.JoinTable(2, 4)
    local d7 = d6 - cudnn.SpatialFullConvolution(ngf*17, ngf*8, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*8) - cudnn.ReLU(true)  -- 16
    
    local d7_16 = {input, d7} - nn.JoinTable(2, 4)
    local d7_32 = d7_16 - cudnn.SpatialFullConvolution(ngf*8+3, ngf*4, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - cudnn.ReLU(true)  -- 32
    
    num_features = 64
    -- Initial processing of the image
	  local conv1 = conv(ngf*4, ngf*2, 3, 3, 1, 1,1,1)(d7_32)
    local r1 = Residual(ngf*2, num_features)(conv1)

    local out = {}
    local inter = r1

    for i = 1, opts.nStack do
        
        local hg = hourglass(4, num_features)(inter)

        -- Residual layers at output resolution
        local ll = Residual(num_features, num_features)(hg) 

        -- Linear layer to produce first set of predictions
        local LL = lin(num_features, num_features, ll)

        -- Predicted heatmaps
        local tmpOut = cudnn.SpatialConvolution(num_features, opts.nOutChannels, 1,1,1,1,0,0)(LL)
        table.insert(out,tmpOut)

        -- Add predictions back
        if i < opt.nStack then
            local ll_ = cudnn.SpatialConvolution(num_features, num_features,1,1,1,1,0,0)(LL)
            local tmpOut_ = cudnn.SpatialConvolution(opts.nOutChannels, num_features,1,1,1,1,0,0)(tmpOut)
            inter = nn.CAddTable()({inter, ll_, tmpOut_})
        end
    end
    
    local d7_32_cat = {d7_32, out[opts.nStack]} - nn.JoinTable(2, 4)
    local d8 = d7_32_cat - cudnn.SpatialConvolution(ngf*4 + opts.nOutChannels, ngf* 2, 3, 3, 1, 1, 1, 1) - Residual(ngf*2, ngf*2) -cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2)  -- 64
    
    local d9 = d8 - cudnn.SpatialConvolution(ngf*2, ngf*2, 3, 3, 1, 1, 1, 1)- Residual(ngf*2, ngf*2) - cudnn.ReLU(true)
    
    local d10 = input - nn.SpatialUpSamplingNearest(4)
    
    local d11 = {d10, d9} - nn.JoinTable(2, 4)

    local d12 = d11 - stn_L2(ngf*2+3) - cudnn.SpatialConvolution(ngf*2+3, ngf*2, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(ngf*2) - cudnn.ReLU(true)  - nn.SpatialUpSamplingNearest(2)  -- 128
    
    local d13 = d12 - cudnn.SpatialConvolution(ngf*2, ngf*2, 3, 3, 1, 1, 1, 1) - Residual(ngf*2, ngf) - Residual(ngf, ngf) - cudnn.ReLU(true)
    local d14 = d13 - cudnn.SpatialConvolution(ngf, ngf/2, 3, 3, 1, 1, 1, 1) - Residual(ngf/2, ngf/2) - cudnn.ReLU(true) - cudnn.SpatialConvolution(ngf/2, 3, 5, 5, 1, 1, 2, 2)

    table.insert(out, d14)

    -- Final model
    local model = nn.gModule({input}, out)


    return model
end


function Model.createNetG_skip_128(opts)

    local ngf = 32
    local input = nn.Identity()()

    local e1 = input - cudnn.SpatialConvolution(3, ngf, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*1)  - nn.LeakyReLU(0.2, true)   -- 8
    local e2 = e1  - cudnn.SpatialConvolution(ngf, ngf*4, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - nn.LeakyReLU(0.2, true)  -- 4
    local e3 = e2  - cudnn.SpatialConvolution(ngf*4, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16)  - nn.LeakyReLU(0.2, true)-- 2
    local e4 = e3  - cudnn.SpatialConvolution(ngf*16, ngf*64, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*64) - nn.LeakyReLU(0.2, true)  -- 1

    local d1 = e4 - cudnn.SpatialFullConvolution(ngf*64, ngf*32, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*32) - cudnn.ReLU(true)  -- 2
    local d2 = {d1, e3} - nn.JoinTable(2, 4)
    local d3 = d2 - cudnn.SpatialFullConvolution(ngf*48, ngf*24, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*24) - cudnn.ReLU(true)  -- 4
    local d4 = {d3, e2} - nn.JoinTable(2, 4)
    local d5 = d4 - cudnn.SpatialFullConvolution(ngf*28, ngf*16, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*16) - cudnn.ReLU(true)  -- 8
    local d6 = {d5, e1} - nn.JoinTable(2, 4)
    local d7 = d6 - cudnn.SpatialFullConvolution(ngf*17, ngf*8, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*8) - cudnn.ReLU(true)  -- 16
    
    local d7_16 = {input, d7} - nn.JoinTable(2, 4)
    local d7_32 = d7_16 - cudnn.SpatialFullConvolution(ngf*8+3, ngf*4, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*4) - cudnn.ReLU(true)  -- 32
    local d7_b1 = d7_32 - cudnn.SpatialFullConvolution(ngf*4, ngf*2, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(ngf*2) - cudnn.ReLU(true) -- 64

    local d7_b2 = input - nn.SpatialUpSamplingNearest(4)
    local d8 = {d7_b2, d7_b1} - nn.JoinTable(2, 4)

    local d9 = d8 - stn_L2(ngf*2+3) - cudnn.SpatialConvolution(ngf*2+3, ngf*2, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(ngf*2) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2)  -- 128
    
    num_features = 64
    -- Initial processing of the image
	  local conv1 = conv(ngf*2, ngf*2, 3, 3, 1, 1,1,1)(d9)
    local r1 = Residual(ngf*2, num_features)(conv1)

    local out = {}
    local inter = r1

    for i = 1, opts.nStack do
        
        local hg = hourglass(4, num_features)(inter)

        -- Residual layers at output resolution
        local ll = Residual(num_features, num_features)(hg) 

        -- Linear layer to produce first set of predictions
        local LL = lin(num_features, num_features, ll)

        -- Predicted heatmaps
        local tmpOut = cudnn.SpatialConvolution(num_features, opts.nOutChannels, 1,1,1,1,0,0)(LL)
        table.insert(out,tmpOut)

        -- Add predictions back
        if i < opt.nStack then
            local ll_ = cudnn.SpatialConvolution(num_features, num_features,1,1,1,1,0,0)(LL)
            local tmpOut_ = cudnn.SpatialConvolution(opts.nOutChannels, num_features,1,1,1,1,0,0)(tmpOut)
            inter = nn.CAddTable()({inter, ll_, tmpOut_})
        end
    end
    
    local d11_ = {d9, out[opts.nStack]} - nn.JoinTable(2, 4)
    local d12 = d11_ - cudnn.SpatialConvolution(ngf*2 + opts.nOutChannels, ngf* 2, 3, 3, 1, 1, 1, 1) - Residual(ngf*2, ngf) -cudnn.ReLU(true) 
    local d13 = d12 - cudnn.SpatialConvolution(ngf, ngf/2, 3, 3, 1, 1, 1, 1) - Residual(ngf/2, ngf/2) - cudnn.ReLU(true) - cudnn.SpatialConvolution(ngf/2, 3, 5, 5, 1, 1, 2, 2)

    table.insert(out, d13)

    -- Final model
    local model = nn.gModule({input}, out)


    return model
end


function Model.createNetG_org(opts)
    model_G = nn.Sequential()
    model_G:add(cudnn.SpatialConvolution(3, 512, 3, 3, 1, 1, 1, 1))
    model_G:add(cudnn.SpatialBatchNormalization(512))
    model_G:add(cudnn.ReLU(true))  
    model_G:add(nn.SpatialUpSamplingNearest(2))  
    model_G:add(stn_L1(512))  
    model_G:add(cudnn.SpatialConvolution(512,256, 3, 3, 1, 1, 1, 1))
    model_G:add(cudnn.SpatialBatchNormalization(256))
    model_G:add(cudnn.ReLU(true))  
    model_G:add(nn.SpatialUpSamplingNearest(2))
    model_G:add(stn_L2(256))    
    model_G:add(cudnn.SpatialConvolution(256, 128, 5, 5, 1, 1, 2, 2))
    model_G:add(cudnn.SpatialBatchNormalization(128))
    model_G:add(cudnn.ReLU(true))
    model_G:add(nn.SpatialUpSamplingNearest(2))
    --model_G:add(stn_L3)    
    model_G:add(cudnn.SpatialConvolution(128, 64, 5, 5, 1, 1, 2, 2))  
    model_G:add(cudnn.SpatialBatchNormalization(64))
    model_G:add(cudnn.ReLU(true))

    model_G:add(cudnn.SpatialConvolution(64, 3, 5, 5, 1, 1, 2, 2))
    return model_G
end


function Model.createNetG_cvpr_lm(opts)

    local ngf = 32
    local input = nn.Identity()()
   
    local d7 = input - cudnn.SpatialConvolution(3, 512, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(512) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2)

    local d8 = d7 - cudnn.SpatialConvolution(512,256, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(256) - cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2)
    
    local d9 = d8 - stn_L2(256) - cudnn.SpatialConvolution(256, 128, 5, 5, 1, 1, 2, 2) - cudnn.SpatialBatchNormalization(128) - cudnn.ReLU(true)
    num_features = 64
    -- Initial processing of the image
	  local conv1 = conv(128, ngf*2, 3, 3, 1, 1,1,1)(d9)
    local r1 = Residual(ngf*2, num_features)(conv1)

    local out = {}
    local inter = r1

    for i = 1, opts.nStack do
        
        local hg = hourglass(4, num_features)(inter)

        -- Residual layers at output resolution
        local ll = Residual(num_features, num_features)(hg) 

        -- Linear layer to produce first set of predictions
        local LL = lin(num_features, num_features, ll)

        -- Predicted heatmaps
        local tmpOut = cudnn.SpatialConvolution(num_features, opts.nOutChannels, 1,1,1,1,0,0)(LL)
        table.insert(out,tmpOut)

        -- Add predictions back
        if i < opt.nStack then
            local ll_ = cudnn.SpatialConvolution(num_features, num_features,1,1,1,1,0,0)(LL)
            local tmpOut_ = cudnn.SpatialConvolution(opts.nOutChannels, num_features,1,1,1,1,0,0)(tmpOut)
            inter = nn.CAddTable()({inter, ll_, tmpOut_})
        end
    end

    local d11_ = {d9, out[opts.nStack]} - nn.JoinTable(2, 4)
    local d12 = d11_ - cudnn.SpatialConvolution(128 + opts.nOutChannels, ngf* 2, 3, 3, 1, 1, 1, 1) - Residual(ngf*2, ngf*2) -cudnn.ReLU(true) - nn.SpatialUpSamplingNearest(2) 
    local d12_b1 = d12 - cudnn.SpatialConvolution(ngf*2, ngf, 3, 3, 1, 1, 1, 1)- Residual(ngf, ngf) - cudnn.ReLU(true)
    local d12_b2 = d9 - nn.SpatialUpSamplingNearest(2) - cudnn.SpatialConvolution(ngf*2, ngf*2, 3, 3, 1, 1, 1, 1) - Residual(ngf*2, ngf*2) - cudnn.ReLU(true)
    
    local d12_ = {d12_b2, d12_b1} - nn.JoinTable(2, 4)
    
    local d13 = d12_ - cudnn.SpatialConvolution(ngf*3, ngf*2, 3, 3, 1, 1, 1, 1) - Residual(ngf*2, ngf) - Residual(ngf, ngf) - cudnn.ReLU(true)
    local d14 = d13 - cudnn.SpatialConvolution(ngf, ngf/2, 3, 3, 1, 1, 1, 1) - Residual(ngf/2, ngf/2) - cudnn.ReLU(true) - cudnn.SpatialConvolution(ngf/2, 3, 5, 5, 1, 1, 2, 2)

    table.insert(out, d14)

    -- Final model
    local model = nn.gModule({input}, out)


    return model
end


return Model