open FSom
open System.Linq
open System.Collections.Generic
open System
open System.Diagnostics

let stopWatch = Stopwatch()

let tic () = 
    stopWatch.Restart()

let toc () = 
        stopWatch.Stop()
        stopWatch.Elapsed.TotalMilliseconds

let mainTainTest argv = 
    let bound = 12000
    let nodes = ([1..bound] |> Seq.map (fun i -> Node(12))).ToList()
    let som1 = SomGpu((200, 200), nodes)

    printfn "training with massive iterations bmu\n"
    som1.Train 5 |> ignore

    0

//[<EntryPoint>]
let main argv = 
    
    for j = 3 to 3 do
        let upper = 10. ** float(j)
        let bound = int(floor(upper * 1.2))
        let nodes = ([1..bound] |> Seq.map (fun i -> Node(12))).ToList()
        let som1 = SomGpu((200, 200), nodes)

        printfn "\n"
        printfn "Number of nodes: %d" bound
        for i = 1 to 3 do
            printfn "\n"
            printfn "============================"
            printfn "\tAttempt: %d" i

            tic()
            let minsGpu2 = som1.GetBmuGpu nodes
            printfn "\tgpu iterations multiple copies of nodes: %10.3f ms" (toc())

            tic()
            let minsGpu1 = som1.GetBmuGpuNodeByNode nodes 
            printfn "\tgpu node-by-node: %10.3f ms" (toc())


            if j < 2 then
                tic()
                nodes |> Seq.iter( fun node -> som1.GetBMUParallel(node) |> ignore)
                printfn "\tcpu parallel: %10.3f ms" (toc())

                tic()
                nodes |> Seq.iter( fun node -> som1.GetBMU(node) |> ignore)
                printfn "\tcpu sequential: %10.3f ms" (toc())

                let rnd = Random(int(DateTime.Now.Ticks))
                let ind = rnd.Next(0, bound)
                tic()
                let min = som1.GetBMU(nodes.[ind])
                printfn "cpu timing for a single node: %10.3f ms" (toc())

let mainBmuTest argv = 
    let bound = 1200
    let dim1 = 30
    let dim2 = 20
    let nodes = ([1..bound] |> Seq.map (fun i -> Node(11))).ToList()
    let som1 = SomGpu((dim1, dim2), nodes)
    let bmus = som1.GetBmuGpuUnified nodes
    let mutable failed = 0
    for i = 0 to bound - 1 do
        let bmu = som1.GetBMU nodes.[i]
        let bmuSom = som1.toSomCoordinates bmus.[i]
        
        if bmu = bmuSom then 
            () //printfn "Success!"
        else
            printfn "ind: %d, bmu: %A, bmuSom: %A" i bmu bmuSom
            failed <- failed + 1
    printfn "failed: %d" failed

let classifyTest argv = 
    let bound = 52000
    let dim1 = 8
    let dim2 = 8
    let nodes = ([1..bound] |> Seq.map (fun i -> Node(12))).ToList()
    let som1 = SomGpu((dim1, dim2), nodes)
    for i = 1 to 3 do
        printfn "\n"
        printfn "============================"
        printfn "\tAttempt: %d" i

        tic()
        let minsGpu1 = som1.GetBmuGpu nodes 
        printfn "\tgpu node-by-node: %10.3f ms" (toc())

let findDistance argv =
    let bound = 52000
    let dim1 = 200
    let dim2 = 200
    let nodes = ([1..bound] |> Seq.map (fun i -> Node(12))).ToList()
    let som1 = SomGpu((dim1, dim2), nodes)
    
    for i = 1 to 3 do
        printfn "\n"
        printfn "============================"
        printfn "\tAttempt: %d" i

        tic()
        let distGpu = som1.DistanceMap() 
        printfn "\tgpu distance map: %10.3f ms" (toc())

        tic()
        let dist = som1.LinearGetDistanceMap()
        printfn "\tlinear distance map: %10.3f ms" (toc())

        let rnd = Random(int(DateTime.Now.Ticks))
        let i = rnd.Next(0, dim1)
        let j = rnd.Next(0, dim2)

        printfn "\tlinear[%d, %d]=%10.5f, gpu[%d, %d]=%10.5f" i j dist.[i,j] i j distGpu.[i,j]


[<EntryPoint>]
let tests argv =
    //classifyTest argv
    //mainTainTest argv
    //for i = 0 to 10 do
    mainBmuTest argv
    //main argv
    //findDistance argv
    0