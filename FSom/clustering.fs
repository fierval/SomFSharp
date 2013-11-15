namespace FSom

open MathNet.Numerics.LinearAlgebra.Double
open System.Threading
open System.Threading.Tasks
open System.Collections.Generic
open System.Linq
open Microsoft.FSharp.Collections
open System
open System.Diagnostics

[<AutoOpen>]
module Clustering =
   
    type Labels =
    | Unlabeled = -1
    | Watershed = 0

    type Direction =
    | Ascend = 0
    | Descend = 1

    let (|Ascend|Descend|) = function
        |Direction.Ascend -> Ascend
        |Direction.Descend -> Descend
        |_ -> failwith "Impossible direction"
        
    [<DebuggerDisplay("({xy})")>]
    type Point =
    |Point of X : int * Y : int

        member this.xy = 
            match this with
            | Point(X, Y) -> X, Y
        
        member this.x =
            match this with
            |Point(X = x) -> x

        member this.y =
            match this with
            |Point(Y = y) -> y

        //neighbors of a point on a rectangular grid
        member this.getNeighbors width height =
            let row, col = this.xy
            let neighbors = List<Point>()
            for i = -1 to 1 do
                for j = -1 to 1 do
                    if (i <> 0 || j <> 0) && i + row >= 0 && j + col >= 0 && i + row < height && j + col < width then
                        neighbors.Add(Point(row + i, col + j))
            neighbors

        member this.addNeighbors width height =
            let neighbors = this.getNeighbors width height
            neighbors.Add(this)
            neighbors

        override this.ToString () =
            this.x.ToString() + "," + this.y.ToString()

       static member Copy (point : Point) = Point(point.xy)

    type AscentDescent(uMatrix : float[,], pMatrix : int [,]) =
        let uMatrix = uMatrix
        let pMatrix = pMatrix |> Array2D.map (fun e -> float e)
        let width = uMatrix |> Array2D.length2
        let height = uMatrix |> Array2D.length1
        
        [<DefaultValue>] val mutable Immersion : Point[,]

        // ascent/descent parameters: minDistance between points
        // where the process should stop
        let precision = 0.00001

        //max number of iterations after which the process stops
        let maxIterations = (width + height) / 2

        // move around the matrix whether ascending or descending
        let ascendOrDescend (point : Point) (direction : Direction) =
            let mutable stop = false
            let mutable movingPoint = Point.Copy(point)
            let mutable iteration = 0
            let mutable currentMinMax = movingPoint
            let m =
                match direction with
                | Ascend -> pMatrix
                | Descend -> uMatrix

            while not stop do
                // the point and all its neighbors
                let neighbors = movingPoint.addNeighbors width height
                currentMinMax <-
                    neighbors
                    |> 
                    match direction with
                    |Ascend ->
                        Seq.maxBy
                            (fun p ->
                                let x, y = p.xy
                                m.[x,y]
                                )
                    |Descend ->
                        Seq.minBy
                            (fun p ->
                                let x, y = p.xy
                                m.[x,y]
                                )
                
                // if minimal point is the point we started from - short circuit                
                if currentMinMax = movingPoint 
                    || abs (m.[movingPoint.x, movingPoint.y] - m.[currentMinMax.x, currentMinMax.y]) < precision 
                    || iteration >= maxIterations
                then stop <- true

                iteration <- iteration + 1
                movingPoint <- currentMinMax

            currentMinMax

        // produces the immersion matrix
        member this.Immerse () =
            // descent matrix maps all points to their "minimums"
            let descent = 
                Array2D.init height width
                    (fun i j -> ascendOrDescend (Point(i, j)) Direction.Descend)
            this.Immersion <-
                Array2D.init height width
                    (fun i j -> ascendOrDescend descent.[i,j] Direction.Ascend)
                                  
    type Watershed (uMatrix: float [,]) =
        let uMatrix = uMatrix
        let width = uMatrix |> Array2D.length2
        let height = uMatrix |> Array2D.length1

        let labels = Array2D.init height width (fun i j -> int Labels.Unlabeled)
        let minimalpoints = Array2D.zeroCreate height width
        let localmins = Array.zeroCreate (height * width) 
        let mutable basins = 0

        /// given a u-matrix retrun a map of its heights
        let getOrderedHeights () =
            let heightMap = Dictionary<float, System.Collections.Generic.List<int*int>>()
            uMatrix 
            |> Array2D.iteri
                (fun i j h -> 
                    if not (heightMap.ContainsKey h) then 
                        heightMap.Add(h, List<int*int>())
                    let lst = heightMap.[h]
                    lst.Add(i,j)
                )
            heightMap.OrderBy(fun kvp -> kvp.Key)

        let processShed (point : Point) =
            let neighbors = point.getNeighbors width height
            
            let availableLabels = HashSet(HashIdentity.Structural)

            let mutable minNeighbor = Unchecked.defaultof<Point>
            let mutable minNeighborHeight = Double.MaxValue
            let mutable nUnlabeled = 0
            let mutable nBorders = 0

            for neighbor in neighbors do
                let row, col = neighbor.xy
                let marker = labels.[row, col]
                let height = uMatrix.[row, col]
                if marker <> int Labels.Unlabeled && height < minNeighborHeight then
                    minNeighbor <- neighbor
                    minNeighborHeight <- height

                match marker with 
                | m when m = int Labels.Unlabeled -> nUnlabeled <- nUnlabeled + 1
                | n when n = int Labels.Watershed -> nBorders <- nBorders + 1
                | _ -> availableLabels.Add(marker)
            
            let row, col = point.xy
            if availableLabels.Count = 0 then
                if nUnlabeled > 0 then
                    // new basin
                    basins <- basins + 1
                    labels.[row, col] <- basins
                    minimalpoints.[row, col] <- point
                    localmins.[basins] <- point
                else
                    // watershed
                    labels.[row, col] <- int Labels.Watershed
                    minimalpoints.[row, col] <- minNeighbor
            elif availableLabels.Count = 1 then
                // belong to an old basin
                let marker = availableLabels.First()
                labels.[row, col] <- marker
                minimalpoints.[row, col] <- minNeighbor
            else
                // more than one label, hence - watershed!
                labels.[row, col] <- int Labels.Watershed
                
                
        /// Actually create the watershed
        member this.CreateWatersheds () =
            let heightMap = getOrderedHeights()
            heightMap.Select(
                    fun kvp -> 
                        let points = kvp.Value
                        points |> Seq.iter (fun p -> processShed (Point(p)))
            ) |> ignore
   
        member this.Labels = labels